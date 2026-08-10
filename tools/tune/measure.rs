//! Robust timing, interleaved A/B comparisons, and noise calibration.
//!
//! # Two-quality probing
//!
//! Crossover sweeps spend most of their probes either walking the ladder
//! looking for the first win or bisecting the transition band. Both only need
//! a directional answer, so they run at [`ProbeQuality::Coarse`]: three
//! alternating paired slots and a quarter of the precise batch. Only the win that starts
//! a bisection, the final confirmation, and the guard cell run at
//! [`ProbeQuality::Precise`] — the exact parameters the tuner used for every
//! probe before — so the accepted crossover keeps its precision while the
//! search around it becomes roughly an order of magnitude cheaper.

use core::{cmp::max, time::Duration};
use std::time::Instant;

/// How much timing a probe invests in its answer.
///
/// Precise is the full measurement: nine paired slots at the full batch.
/// Coarse is a directional answer only: three slots at a quarter batch, used
/// where the outcome is revisited by a later precise probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeQuality {
    Coarse,
    Precise,
}

impl ProbeQuality {
    /// Number of alternating paired slots for this quality.
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

/// Return a configurable odd-sample median.
///
/// Compile-time SSA candidates include RAM-sized cells whose individual runs
/// take seconds. They use fewer samples than crossover cells while retaining a
/// median, so one noisy scheduling event cannot select a profile.
pub fn median_batch_samples<F: FnMut()>(
    mut operation: F,
    iterations: u32,
    sample_count: usize,
) -> Duration {
    assert!(
        sample_count > 0 && !sample_count.is_multiple_of(2),
        "median sample count must be positive and odd"
    );
    let warmup_iterations = iterations.checked_div(4).unwrap_or(0);
    for _ in 0..warmup_iterations {
        operation();
    }

    let mut samples = vec![Duration::MAX; sample_count];
    for sample in &mut samples {
        let started = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        *sample = started.elapsed();
    }
    median_of(samples)
}

/// Compare two operations with paired, symmetric interleaved sampling.
///
/// Even slots run A/B/B/A and odd slots B/A/A/B. Summing each operation's two
/// batches inside the same slot cancels slow frequency drift; alternating the
/// outer operation prevents a persistent first/last-position advantage. The
/// returned durations are the medians of those paired slot totals.
pub fn median_pair_batches(
    mut baseline: impl FnMut(),
    mut candidate: impl FnMut(),
    iterations: u32,
    sample_count: usize,
) -> (Duration, Duration) {
    assert!(
        sample_count > 0 && !sample_count.is_multiple_of(2),
        "median sample count must be positive and odd"
    );
    let warmup_iterations = iterations.checked_div(4).unwrap_or(0);
    for _ in 0..warmup_iterations {
        baseline();
        candidate();
    }

    let mut baseline_samples = Vec::with_capacity(sample_count);
    let mut candidate_samples = Vec::with_capacity(sample_count);
    for slot in 0..sample_count {
        let (baseline_first, candidate_first, candidate_second, baseline_second) =
            if slot.is_multiple_of(2) {
                (
                    time_batch(&mut baseline, iterations),
                    time_batch(&mut candidate, iterations),
                    time_batch(&mut candidate, iterations),
                    time_batch(&mut baseline, iterations),
                )
            } else {
                let candidate_outer_first = time_batch(&mut candidate, iterations);
                let baseline_inner_first = time_batch(&mut baseline, iterations);
                let baseline_inner_second = time_batch(&mut baseline, iterations);
                let candidate_outer_second = time_batch(&mut candidate, iterations);
                (
                    baseline_inner_first,
                    candidate_outer_first,
                    candidate_outer_second,
                    baseline_inner_second,
                )
            };
        baseline_samples.push(baseline_first.saturating_add(baseline_second));
        candidate_samples.push(candidate_first.saturating_add(candidate_second));
    }
    (median_of(baseline_samples), median_of(candidate_samples))
}

fn time_batch(operation: &mut impl FnMut(), iterations: u32) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    started.elapsed()
}

/// Acceptance margin in parts per million, from a measured host noise level.
///
/// The floor is 2%: below that, crossover differences are not actionable on
/// any machine this tuner is meant to run on. Above the floor, the margin
/// tracks three standard deviations of the host's own coefficient of variation
/// (measured by [`crate::crossovers::calibrate_noise`]), so a host whose
/// timing spreads 10% will not accept a 2% difference as a win.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the noise coefficient is a bounded non-negative fraction; the ppm integer is clamped to [2%, 15%] after conversion"
)]
pub fn acceptance_margin(noise_cv: f64) -> u32 {
    let sigma_ppm = (3.0 * noise_cv * 1_000_000.0).round() as u32;
    sigma_ppm.clamp(20_000, 150_000)
}

/// True when `candidate` beats `baseline` by at least `margin_ppm` ppm.
#[must_use]
pub fn confidently_faster_nanos(candidate: u128, baseline: u128, margin_ppm: u32) -> bool {
    let factor = u128::from(1_000_000_u32.wrapping_sub(margin_ppm));
    candidate.saturating_mul(1_000_000) < baseline.saturating_mul(factor)
}

/// Locate the first crossover that remains a win at a later guard point.
///
/// The ladder scan and the bisection run at [`ProbeQuality::Coarse`]: both
/// only need a direction, and every answer they produce is revisited. Once the
/// bisection converges, the candidate is confirmed at [`ProbeQuality::Precise`]
/// — stepping up through the ladder if the coarse search landed inside the
/// host's noise band — and the later precise guard rejects isolated noisy
/// wins. The accepted threshold therefore carries exactly the precision of a
/// full measurement while the search around it runs at a fraction of the cost.
///
/// `wins` should include the desired confidence margin.
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
            let middle = low.wrapping_add(high.wrapping_sub(low).div_euclid(2));
            if wins(middle, ProbeQuality::Coarse) {
                candidate = middle;
                if middle == 0 {
                    break;
                }
                high = middle.wrapping_sub(1);
            } else {
                low = middle.wrapping_add(1);
            }
        }

        // Precise confirmation. The coarse bisection can land one or two
        // steps inside the noise band around the boundary, so step up through
        // the ladder until a precise win is found; three tries cover the
        // noise the calibration margin is derived from.
        let mut confirmed = None;
        for _ in 0..3 {
            if wins(candidate, ProbeQuality::Precise) {
                confirmed = Some(candidate);
                break;
            }
            let next = sizes
                .iter()
                .copied()
                .find(|&next_size| next_size > candidate)
                .unwrap_or_else(|| candidate.saturating_add(1));
            if next == candidate {
                break;
            }
            candidate = next;
        }
        let Some(confirmed_candidate) = confirmed else {
            println!("  - coarse win at {size} did not survive precise confirmation");
            last_loss = candidate;
            continue;
        };

        let guard = confirmed_candidate.wrapping_add(confirmed_candidate.div_euclid(4).max(16));
        if wins(guard, ProbeQuality::Precise) {
            println!("Confirmed {tag} crossover at {confirmed_candidate} limbs");
            return Some(confirmed_candidate);
        }
        last_loss = guard;
    }

    println!("No sustained {tag} crossover was found in the measured range");
    None
}

fn median_of(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples
        .get(samples.len().div_euclid(2))
        .copied()
        .expect("positive odd sample count has a median")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_crossover_finds_the_boundary_with_two_qualities() {
        let mut coarse_probes = 0;
        let mut precise_probes = 0;
        let result = sustained_crossover(
            8,
            &[8, 16, 32, 64, 128, 256, 512, 1_024],
            "test",
            |len, quality| {
                match quality {
                    ProbeQuality::Coarse => coarse_probes += 1,
                    ProbeQuality::Precise => precise_probes += 1,
                }
                len >= 200
            },
        );
        assert_eq!(result, Some(200));
        // Exactly two precise probes: the confirmation and the guard.
        assert_eq!(precise_probes, 2);
        assert!(coarse_probes > precise_probes);
    }

    #[test]
    fn sustained_crossover_rejects_an_isolated_noisy_win() {
        // A one-size spike at 64 and the real boundary at 300.
        let result = sustained_crossover(
            8,
            &[8, 16, 32, 64, 128, 256, 512],
            "test",
            |len, _quality| len == 64 || len >= 300,
        );
        // The spike survives its own guard only if it is sustained; it is not,
        // so the sweep continues and lands on the real boundary.
        assert_eq!(result, Some(300));
    }
}
