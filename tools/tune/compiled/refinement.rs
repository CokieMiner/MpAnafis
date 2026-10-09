//! Local interval refinement and bounded searches of coupled policy pairs.

use super::{CandidateTrial, CompiledTuner, CoordinateSearch, Knob, ScoreDomain, TuneSession};

/// Policy pairs share recursive geometry, admission constraints, or product strategies.
const COUPLED_POLICIES: &[(&str, &str)] = &[
    (
        "RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD",
        "RADIX_PARSE_LEAF_CHUNKS",
    ),
    (
        "RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD",
        "RADIX_PARSE_LEAF_CHUNKS",
    ),
    (
        "RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD",
        "RADIX_PARSE_LEAF_CHUNKS",
    ),
    ("BURNIKEL_ZIEGLER_THRESHOLD", "BURNIKEL_ZIEGLER_BLOCK_LIMBS"),
    ("NEWTON_RAPHSON_THRESHOLD", "NEWTON_RAPHSON_BASECASE_LIMBS"),
    ("BURNIKEL_ZIEGLER_THRESHOLD", "NEWTON_RAPHSON_THRESHOLD"),
    (
        "BURNIKEL_QUOTIENT_THRESHOLD",
        "APPROXIMATE_DIVISION_BLOCK_LIMBS",
    ),
    ("NEWTON_QUOTIENT_THRESHOLD", "NEWTON_RAPHSON_BASECASE_LIMBS"),
    ("BURNIKEL_QUOTIENT_THRESHOLD", "NEWTON_QUOTIENT_THRESHOLD"),
    (
        "DIVISION_TRUNCATION_RATIO",
        "NEWTON_SMALL_QUOTIENT_BLOCK_RATIO",
    ),
    ("SSA_BASE_MODULUS_BITS", "SSA_BNM1_BASECASE_LIMBS"),
    ("MONTGOMERY_CIOS_MAX_LIMBS", "MUL_MOD_BNM1_THRESHOLD"),
    ("SSA_BASE_MODULUS_BITS", "CACHE_BLOCK_BYTES"),
    ("SSA_SHIFT_SCALAR_THRESHOLD", "SSA_SHIFT_BLOCK_WIDTH"),
    (
        "SSA_NEGACYCLIC_FACTOR3_THRESHOLD",
        "SSA_NEGACYCLIC_FACTOR5_THRESHOLD",
    ),
    (
        "SSA_BASECASE_COST_WEIGHT_16THS",
        "SSA_NESTED_COST_PENALTY_16THS",
    ),
    ("TRANSFORM_MIN_SMALLER_LIMBS", "TRANSFORM_MAX_OPERAND_RATIO"),
    ("HGCD_BLOCK_THRESHOLD", "HGCD_CROSSOVER_THRESHOLD"),
    (
        "EXTENDED_HGCD_CROSSOVER_THRESHOLD",
        "EXTENDED_GCD_WIDE_THRESHOLD",
    ),
    (
        "EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS",
        "EXTENDED_GCD_COFACTOR_BATCH_RATIO",
    ),
    (
        "SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD",
        "SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS",
    ),
    (
        "SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD",
        "SSA_PARALLEL_MIN_LIMB_WORK",
    ),
];

impl CoordinateSearch {
    /// Explore a local Cartesian grid after each coordinate pass. A paired
    /// update can satisfy ordering constraints or improve a coupled policy
    /// even when neither individual change can be accepted.
    pub fn tune_pairs(session: &mut TuneSession, knobs: &[Knob<'_>]) -> bool {
        let mut changed = false;
        for &(first, second) in COUPLED_POLICIES {
            let (Some(&a), Some(&b)) = (
                knobs.iter().find(|knob| knob.name == first),
                knobs.iter().find(|knob| knob.name == second),
            ) else {
                continue;
            };
            if a.domain != b.domain {
                continue;
            }
            let pairs = [a, b];
            let current_a = (a.get)(session.profile);
            let current_b = (b.get)(session.profile);
            let mut left_values = Self::pair_values(&a, current_a);
            let mut right_values = Self::pair_values(&b, current_b);
            if current_a == 0 || current_b == 0 {
                // A disabled strategy masks its partner's timing effect. The
                // declared joint grid must include an enabled strategy with
                // admissible geometry or worker budgets in the same profile.
                left_values.extend_from_slice(a.candidates);
                right_values.extend_from_slice(b.candidates);
                left_values.sort_unstable();
                left_values.dedup();
                right_values.sort_unstable();
                right_values.dedup();
            }
            let mut trials = Vec::new();
            for left in left_values {
                for &right in &right_values {
                    let mut profile = session.profile;
                    (a.set)(&mut profile, left);
                    (b.set)(&mut profile, right);
                    if profile != session.profile && profile.validate().is_ok() {
                        trials.push(CandidateTrial {
                            profile,
                            values: vec![left, right],
                        });
                    }
                }
            }
            if trials.is_empty() {
                continue;
            }
            if a.domain == ScoreDomain::Parsing {
                let mut profiles = vec![session.profile];
                profiles.extend(trials.iter().map(|trial| trial.profile));
                session.harness.parsing_chunks = CompiledTuner::parsing_grid(&profiles, true);
            }
            if a.domain == ScoreDomain::ProductionDivision {
                let mut profiles = vec![session.profile];
                profiles.extend(trials.iter().map(|trial| trial.profile));
                session.harness.division_grid = CompiledTuner::division_grid(&profiles, true);
            }
            if a.domain == ScoreDomain::Products {
                let mut profiles = vec![session.profile];
                profiles.extend(trials.iter().map(|trial| trial.profile));
                session.harness.product_grid = CompiledTuner::product_grid(&profiles, true);
            }
            println!(
                "\nCoupled search {first}, {second}: {} valid profiles",
                trials.len()
            );
            let plan = Self::measurement_plan(session, &pairs);
            let Some(mut screened) = Self::screen_trials(session, a.domain, &plan, &trials) else {
                return changed;
            };
            changed |= Self::confirm_trials(session, &pairs, &plan, &trials, &mut screened);
            if !session.harness.check_context() {
                return changed;
            }
        }
        changed
    }

    /// Integer neighbours and both half-intervals of promising anchors.
    /// Refinement stays within the declared grid and incumbent envelope;
    /// profile validation subsequently enforces mathematical constraints.
    #[must_use]
    pub fn refine_candidates(knob: &Knob<'_>, current: usize, anchors: &[usize]) -> Vec<usize> {
        let mut grid = knob.candidates.to_vec();
        grid.push(current);
        grid.sort_unstable();
        grid.dedup();
        let minimum = *grid.first().expect("coordinate grid includes incumbent");
        let maximum = *grid.last().expect("coordinate grid includes incumbent");
        let mut refined = Vec::new();
        for &anchor in anchors {
            if anchor == 0 {
                continue;
            }
            if let Some(value) = anchor.checked_sub(1).filter(|&value| value >= minimum) {
                refined.push(value);
            }
            if let Some(value) = anchor.checked_add(1).filter(|&value| value <= maximum) {
                refined.push(value);
            }
            if let Some(&lower) = grid.iter().rev().find(|&&value| value < anchor) {
                refined.push(
                    lower
                        .checked_add(
                            anchor
                                .checked_sub(lower)
                                .expect("ordered interval")
                                .div_euclid(2),
                        )
                        .expect("midpoint within interval"),
                );
            }
            if let Some(&upper) = grid.iter().find(|&&value| value > anchor) {
                refined.push(
                    anchor
                        .checked_add(
                            upper
                                .checked_sub(anchor)
                                .expect("ordered interval")
                                .div_ceil(2),
                        )
                        .expect("midpoint within interval"),
                );
            }
        }
        refined.sort_unstable();
        refined.dedup();
        refined
    }

    /// Keep the incumbent, adjacent declared grid values, and integer neighbours.
    /// This finite local search makes no monotonicity assumption about timings.
    #[must_use]
    pub fn pair_values(knob: &Knob<'_>, current: usize) -> Vec<usize> {
        let mut values = vec![current];
        if let Some(&lower) = knob
            .candidates
            .iter()
            .filter(|&&value| value < current)
            .max()
        {
            values.push(lower);
        }
        if let Some(&upper) = knob
            .candidates
            .iter()
            .filter(|&&value| value > current)
            .min()
        {
            values.push(upper);
        }
        values.extend(
            Self::refine_candidates(knob, current, &[current])
                .into_iter()
                .filter(|&value| current.abs_diff(value) == 1),
        );
        values.sort_unstable();
        values.dedup();
        values
    }
}
