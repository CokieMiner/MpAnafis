//! Sustained crossover search, probe quality, and noise margin bounds.

use core::cmp::max;

/// Minimum relative improvement required for crossover acceptance.
pub const MIN_MARGIN_PPM: u32 = 7_500; // 0.75%

/// Maximum relative improvement margin derived from calibration noise.
pub const MAX_MARGIN_PPM: u32 = 50_000; // 5.0%

/// At most nine local limb widths are confirmed around a screened boundary.
const CROSSOVER_NEIGHBOUR_RADIUS: usize = 4;

/// Sampling budget for screening or fresh confirmation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeQuality {
    /// Directional three-slot probe at a quarter of the precise batch.
    Coarse,
    /// Pilot and fresh confirmation at the precise batch.
    Precise,
}

impl ProbeQuality {
    /// Screening or pilot slot count; confirmation is budgeted separately.
    #[must_use]
    pub const fn samples(self) -> usize {
        match self {
            Self::Coarse => 3,
            Self::Precise => 9,
        }
    }

    /// The batch this quality runs at, given the precise batch.
    #[must_use]
    pub fn batch(self, precise_batch: u32) -> u32 {
        match self {
            Self::Precise => precise_batch,
            Self::Coarse => precise_batch.checked_div(4).unwrap_or(1).max(3),
        }
    }
}

/// Namespace for crossover searches, batch scaling, and noise margins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CrossoverMeasure;

impl CrossoverMeasure {
    /// Locate a crossover whose measured tail remains consistently faster.
    ///
    /// Coarse ladder and bisection probes screen a boundary. Precise probes
    /// scan a small integer neighbourhood, then confirm the selected boundary's
    /// immediate successors, a nearby guard, and every remaining ladder point.
    ///
    /// `wins` includes the configured noise margin. This is a sampled-range
    /// condition, not a guarantee between samples or outside the measured ladder.
    pub fn sustained_crossover<F>(
        start_size: usize,
        sizes: &[usize],
        tag: &str,
        mut wins: F,
    ) -> Option<usize>
    where
        F: FnMut(usize, ProbeQuality) -> bool,
    {
        let mut last_loss = start_size;
        for &size in sizes {
            if size <= last_loss {
                continue;
            }
            if !wins(size, ProbeQuality::Coarse) {
                last_loss = size;
                println!("  - tested {size}, new algorithm is slower");
                continue;
            }

            let mut low = last_loss;
            let mut high = size;
            let mut candidate = size;
            while low <= high {
                let middle = low.checked_add(high.checked_sub(low)?.div_euclid(2))?;
                if wins(middle, ProbeQuality::Coarse) {
                    candidate = middle;
                    if middle == 0 {
                        break;
                    }
                    high = middle.checked_sub(1)?;
                } else {
                    let Some(next) = middle.checked_add(1) else {
                        break;
                    };
                    low = next;
                }
            }

            // Inspect every integer in the local window, even after a loss.
            // The first win of its final uninterrupted suffix is the boundary;
            // a coarse noisy loss can therefore move it in either direction.
            let first = candidate
                .saturating_sub(CROSSOVER_NEIGHBOUR_RADIUS)
                .max(start_size)
                .max(1);
            let last = candidate.saturating_add(CROSSOVER_NEIGHBOUR_RADIUS);
            let mut confirmed = None;
            for len in first..=last {
                if wins(len, ProbeQuality::Precise) {
                    if confirmed.is_none() {
                        confirmed = Some(len);
                    }
                } else {
                    confirmed = None;
                }
            }
            let Some(confirmed_candidate) = confirmed else {
                println!("  - coarse win at {size} did not survive precise confirmation");
                last_loss = last;
                continue;
            };

            let guard =
                confirmed_candidate.saturating_add(confirmed_candidate.div_euclid(4).max(16));
            let mut guards: Vec<_> = sizes
                .iter()
                .copied()
                .filter(|&len| len > confirmed_candidate)
                .collect();
            guards.push(guard);
            for distance in 1..=CROSSOVER_NEIGHBOUR_RADIUS {
                if let Some(len) = confirmed_candidate.checked_add(distance) {
                    guards.push(len);
                }
            }
            // The local suffix already confirmed these widths with fresh
            // precise observations; do not repeat them as tail guards.
            guards.retain(|&len| len > last);
            guards.sort_unstable();
            guards.dedup();
            let failed_guard = guards
                .into_iter()
                .find(|&len| !wins(len, ProbeQuality::Precise));
            if let Some(failed) = failed_guard {
                last_loss = failed;
            } else {
                println!("Confirmed {tag} crossover at {confirmed_candidate} limbs");
                return Some(confirmed_candidate);
            }
        }

        println!("No sustained {tag} crossover was found in the measured range");
        None
    }

    /// Precise batch size for a balanced multiplication or squaring probe.
    ///
    /// The batches shrink as the operands grow so that one sample slot stays
    /// bounded in wall time; the precision comes from the median over slots.
    #[must_use]
    pub fn balanced_batch(base: u32, len: usize) -> u32 {
        if len >= 4_096 {
            max(3, base.checked_div(500).unwrap_or(3))
        } else if len >= 512 {
            max(10, base.checked_div(20).unwrap_or(10))
        } else {
            max(10, base)
        }
    }

    /// Acceptance margin in parts per million, from a measured host noise level.
    ///
    /// This is a practical improvement floor, not a confidence interval.
    /// Fresh paired confirmation supplies the uncertainty bound separately;
    /// noisy comparisons remain inconclusive even when this floor is capped.
    #[must_use]
    pub fn acceptance_margin(noise_cv_ppm: u32) -> u32 {
        noise_cv_ppm
            .saturating_add(noise_cv_ppm.div_euclid(20))
            .clamp(MIN_MARGIN_PPM, MAX_MARGIN_PPM)
    }

    /// True when `candidate` beats `baseline` by at least `margin_ppm` ppm.
    #[must_use]
    pub fn confidently_faster_nanos(candidate: u128, baseline: u128, margin_ppm: u32) -> bool {
        if candidate == 0 || baseline == 0 {
            return false;
        }
        let factor = u128::from(1_000_000_u32.saturating_sub(margin_ppm));
        match (
            candidate.checked_mul(1_000_000),
            baseline.checked_mul(factor),
        ) {
            (Some(scaled_candidate), Some(scaled_baseline)) => scaled_candidate < scaled_baseline,
            _ => false,
        }
    }
}
