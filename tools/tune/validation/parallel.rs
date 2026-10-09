//! Production multiplication and squaring gates for each explicit worker budget.

use crate::{platform::Platform, worker::ParallelWorker};

use super::{
    CandidateHarness, SCORE_SCALE, ScoreDomain, TuneSession, VALIDATION_AGGREGATE_TOLERANCE_PPM,
    VALIDATION_FAMILY_TOLERANCE_PPM, Validation,
};

impl Validation {
    /// Validate actual production multiplication and squaring in every selected
    /// pool. Serial production evidence cannot satisfy this separate gate.
    pub fn parallel_production(session: &mut TuneSession) -> bool {
        session.harness.selected_cells.clear();
        let mut widths = Vec::new();
        for profile in [session.defaults, session.profile] {
            for value in [
                profile.ssa,
                profile.sqr_ssa,
                profile.ssa_direct_fermat_parallel_threshold,
                profile.transform_min_smaller_limbs,
            ] {
                if value != 0 && value < usize::MAX - 1 {
                    widths.extend([
                        value.saturating_sub(1).max(1),
                        value,
                        value.checked_add(1).expect("finite product boundary"),
                    ]);
                }
            }
        }
        widths.sort_unstable();
        widths.dedup();
        session.harness.parallel_widths = widths;
        let maximum = Platform::parse_cpu_list(&session.harness.parallel_cpus)
            .expect("validated parallel CPUs")
            .len();
        let (multiplication, squaring) =
            ParallelWorker::cell_catalog(true, &session.harness.parallel_widths);
        let per_pool = multiplication
            .len()
            .checked_add(squaring.len())
            .expect("parallel catalog fits");
        let weights = ScoreDomain::ParallelProduction.cell_weights(&session.harness);
        let mut families = Vec::new();
        for pool in 0..maximum {
            let start = pool.checked_mul(per_pool).expect("parallel pool offset");
            let middle = start
                .checked_add(multiplication.len())
                .expect("parallel square offset");
            let end = start.checked_add(per_pool).expect("parallel pool end");
            families.extend([start..middle, middle..end]);
        }
        let limit = SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM);
        let rule = Self::comparison_rule(&weights, limit, &families);
        let passed = session
            .harness
            .compare_profiles(
                &session.defaults,
                &session.profile,
                ScoreDomain::ParallelProduction,
                &rule,
            )
            .is_some_and(|ratios| {
                let reference = vec![SCORE_SCALE; weights.len()];
                Self::cells_non_regress(&ratios, &reference, weights.len())
                    && CandidateHarness::relative_score(&ratios, &reference, &weights) <= limit
                    && families.iter().all(|span| {
                        CandidateHarness::relative_score(
                            ratios.get(span.clone()).expect("parallel family"),
                            reference.get(span.clone()).expect("reference family"),
                            weights.get(span.clone()).expect("family weights"),
                        ) <= SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM)
                    })
            });
        session.record(
            "PARALLEL_PRODUCTION_VALIDATION",
            format!(
                "cpus={}; workers=1..={maximum}; cells_per_pool={per_pool}; verdict={}",
                session.harness.parallel_cpus,
                if passed {
                    "passed"
                } else {
                    "failed or unresolved"
                }
            ),
        );
        passed
    }
}
