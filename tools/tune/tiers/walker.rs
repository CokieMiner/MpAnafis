//! Stack-based tower walker and candidate tracking for tier crossovers.

use core::{cmp::max, fmt::Debug};

use super::{Parameter, TuneSession, TuningProfile};

/// Inner repetitions for conventional-tower crossover probes.
pub const ITERATIONS: u32 = 5_000;

/// Ladder for Schoolbook -> Karatsuba crossover search.
pub const KARATSUBA_SIZES: [usize; 13] = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96];

/// Ladder for Karatsuba -> Toom-3 crossover search.
///
/// The endpoint bounds search work. Failure to find a crossover in this range
/// provides no conclusion about larger sizes and preserves the input tower.
pub const TOOM3_SIZES: [usize; 16] = [
    24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320,
];

/// Ladder for Toom-4 crossover search.
pub const TOOM4_SIZES: [usize; 16] = [
    64, 80, 96, 112, 128, 160, 192, 208, 224, 256, 288, 320, 384, 512, 640, 768,
];

/// Ladder for Toom-6 crossover search with granular spacing.
pub const TOOM6_SIZES: [usize; 14] = [
    256, 384, 512, 640, 768, 896, 1_024, 1_280, 1_536, 1_792, 2_048, 2_560, 3_072, 4_096,
];

/// Ladder for Toom-8.5 crossover search with granular spacing.
pub const TOOM85_SIZES: [usize; 14] = [
    768, 1_024, 1_280, 1_536, 1_792, 2_048, 2_560, 3_072, 3_584, 4_096, 5_120, 6_144, 7_168, 8_192,
];

/// Extended ladder for late conventional tiers and transforms.
pub const LARGE_SIZES: [usize; 12] = [
    512, 768, 1_024, 1_536, 2_048, 3_072, 4_096, 6_144, 8_192, 12_288, 16_384, 32_768,
];

/// One adjacent-tier candidate and the ladder it may claim.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<A> {
    /// Algorithm that this candidate would install.
    pub algo: A,
    /// Measured widths for this transition.
    pub sizes: &'static [usize],
    /// Earliest start width the next higher tier may search from.
    pub min_next_start: usize,
}

/// Stack-based tier tower walker with shadowing fixup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TowerWalker;

impl TowerWalker {
    /// Walk one tier tower under a frozen scoring profile and return the
    /// measured thresholds.
    ///
    /// Every crossover probe in one pass compiles its worker with
    /// `scoring_profile`, guaranteeing identical recursive child dispatch
    /// throughout the tower. `session.profile` is not read for worker
    /// compilation and is not mutated until the pass is complete.
    ///
    /// A shadowed intermediate tier receives the threshold of the next measured
    /// tier, so the higher tier wins the dispatch check at that exact width. A
    /// shadowed tail remains `usize::MAX - 1` because it has no successor that can
    /// encode the shadowing boundary.
    ///
    /// `on_apply` is called once for every tier after the complete tower
    /// (including shadowing fixup) has been determined. It is not called during
    /// measurement.
    ///
    /// # Panics
    ///
    /// Panics if the internal algorithm stack becomes corrupted or empty.
    pub fn tune_tower<A: Copy + Debug>(
        session: &mut TuneSession,
        scoring_profile: &TuningProfile,
        base_algo: A,
        candidates: &[Candidate<A>],
        mut crossover_fn: impl FnMut(
            &mut TuneSession,
            &TuningProfile,
            A,
            A,
            usize,
            &[usize],
            &str,
            u32,
        ) -> Result<Option<usize>, String>,
        mut on_apply: impl FnMut(&mut TuneSession, usize, usize),
    ) -> Vec<usize> {
        let original = session.profile;
        let mut thresholds = vec![usize::MAX - 1; candidates.len()];
        // Stack of (candidate_index, algorithm, starting_threshold)
        let mut stack: Vec<(isize, A, usize)> = vec![(-1, base_algo, 8)];

        let mut i = 0;
        while let Some(cand) = candidates.get(i) {
            let (reign_idx, reign_algo, reign_start) =
                *stack.last().expect("stack must never be empty");

            let dynamic_tag = format!("{:?} -> {:?}", reign_algo, cand.algo);

            let search_start = max(reign_start, cand.min_next_start);

            match crossover_fn(
                session,
                scoring_profile,
                reign_algo,
                cand.algo,
                search_start,
                cand.sizes,
                &dynamic_tag,
                ITERATIONS,
            ) {
                Ok(Some(c)) => {
                    if c <= reign_start && reign_idx != -1 {
                        println!(
                            "Tower rollback: {dynamic_tag} beat the reigning algorithm before it started; shadowing the previous tier."
                        );
                        let rollback_idx = usize::try_from(reign_idx).unwrap_or(0);
                        if let Some(slot) = thresholds.get_mut(rollback_idx) {
                            *slot = usize::MAX - 1;
                        }
                        let _ = stack.pop();
                        continue;
                    }
                    if let Some(slot) = thresholds.get_mut(i) {
                        *slot = c;
                    }
                    stack.push((isize::try_from(i).unwrap_or(0), cand.algo, c));
                }
                Ok(None) => {
                    session.profile = original;
                    session.record(
                        dynamic_tag,
                        "no confirmed crossover in the measured range; tower retained",
                    );
                    return Vec::new();
                }
                Err(error) => {
                    session.profile = original;
                    session.harness.reject(&error);
                    session.record(dynamic_tag, format!("failed: {error}; tower restored"));
                    return Vec::new();
                }
            }
            i = i.saturating_add(1);
        }

        // Dispatch considers higher conventional tiers after lower ones and keeps
        // walking when their thresholds are equal. Therefore assigning a missing
        // intermediate tier the next real threshold makes it unreachable without
        // carrying a sentinel into an otherwise ordered tower.
        let mut next_measured = None;
        for threshold in thresholds.iter_mut().rev() {
            if *threshold == usize::MAX - 1 {
                if let Some(next) = next_measured {
                    *threshold = next;
                }
            } else {
                next_measured = Some(*threshold);
            }
        }

        for (idx, &threshold) in thresholds.iter().enumerate() {
            on_apply(session, idx, threshold);
        }

        thresholds
    }

    /// Records the installed thresholds after dispatch-gate reconciliation.
    ///
    /// The tower's application callback owns profile mutation. Recording reads
    /// that final profile so conventional tiers shadowed by SSA keep their
    /// sentinel, and the recorded decision matches subsequent worker builds.
    pub fn record_thresholds(session: &mut TuneSession, fields: &[Parameter]) {
        for parameter in fields {
            let slot = (parameter.get)(session.profile);
            session.record(parameter.name, slot.to_string());
        }
    }
}
