//! Tests for compiled-profile candidate shortlisting.

use std::{
    env::{consts::ARCH, temp_dir},
    fs::remove_dir_all,
    process::id as process_id,
};

use crate::{
    crossovers::Calibration,
    harness::ComparisonDecision,
    measure::RatioEstimate,
    worker::{PARSING_RADICES, ParsingWorker},
};

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use super::PARALLEL_KNOBS;
use super::{
    CandidateHarness, CompiledTuner, CoordinateGrid, CoordinateSearch, DIVISION_KNOBS,
    DivisionGrid, DivisionWorker, GCD_KNOBS, Knob, PARSING_KNOBS, PRODUCT_KNOBS, SCORE_SCALE,
    SSA_KNOBS, ScoreDomain, TOOM_KNOBS, TRANSFORM_SHAPE_KNOBS, TuneSession, TuningProfile,
    Validation, plan::MeasurementPlan,
};

#[test]
fn confirmation_preserves_direct_weights_and_leaves_consumer_cells_out_of_the_objective() {
    let plan = MeasurementPlan {
        indices: vec![0, 2, 5],
        weights: vec![1, 2, 1],
    };
    assert_eq!(plan.complete_weights(8), [1, 0, 2, 0, 0, 1, 0, 0]);
    let mut rule = Validation::comparison_rule(&plan.complete_weights(8), SCORE_SCALE - 1, &[]);
    let baseline = vec![SCORE_SCALE; 8];
    let mut ratios = vec![950_000; 8];
    *ratios.get_mut(7).expect("consumer guard") = 1_030_001;
    assert!(!Validation::cells_non_regress(&ratios, &baseline, 8));
    // A faster direct objective cannot compensate for an individual guard.
    rule.cell_maximum = 1_030_000;
    let bounds = ratios
        .into_iter()
        .map(|ratio| RatioEstimate {
            median: ratio,
            lower: ratio,
            upper: ratio,
        })
        .collect::<Vec<_>>();
    assert_eq!(rule.decision(&bounds), ComparisonDecision::Reject);
}

#[test]
fn product_grid_preserves_reference_selected_and_candidate_boundaries() {
    let reference = TuningProfile::portable();
    let selected = TuningProfile {
        montgomery_cios_max_limbs: 17,
        mul_mod_bnm1: 211,
        ..reference
    };
    let grid = CompiledTuner::product_grid(&[reference, selected], true);
    for knob in PRODUCT_KNOBS {
        let widths = if knob.name == "MONTGOMERY_CIOS_MAX_LIMBS" {
            &grid.montgomery
        } else {
            &grid.cyclic
        };
        for value in [reference, selected]
            .map(|profile| (knob.get)(profile))
            .into_iter()
            .chain(knob.candidates.iter().copied())
            .filter(|&value| value > 0)
        {
            for neighbour in [value.saturating_sub(1).max(1), value, value + 1] {
                assert!(widths.contains(&neighbour), "{}: {neighbour}", knob.name);
            }
        }
        assert!(
            widths
                .windows(2)
                .all(|pair| matches!(pair, [a, b] if a < b))
        );
    }
}

#[test]
fn parsing_grids_cover_root_and_leaf_boundaries_with_disjoint_radix_objectives() {
    let reference = TuningProfile::portable();
    let selected = TuningProfile {
        radix_parse_decimal_recursive: 73,
        radix_parse_small_recursive: 91,
        radix_parse_large_recursive: 117,
        radix_parse_leaf: 7,
        ..reference
    };
    let widths = CompiledTuner::parsing_grid(&[reference, selected], false);
    let search_widths = CompiledTuner::parsing_grid(&[selected], true);
    let weights = ParsingWorker::cell_weights(&widths);
    assert!(
        widths
            .windows(2)
            .all(|pair| matches!(pair, [a, b] if a < b))
    );
    assert!([18, 19, 20].iter().all(|width| widths.contains(width)));
    for knob in PARSING_KNOBS {
        assert_eq!(knob.domain, ScoreDomain::Parsing);
        for profile in [reference, selected] {
            let value = (knob.get)(profile);
            for neighbour in [value.saturating_sub(1).max(1), value, value + 1] {
                assert!(widths.contains(&neighbour), "{}: {neighbour}", knob.name);
            }
        }
        for &value in knob.candidates {
            assert!(search_widths.contains(&value));
            let mut candidate = reference;
            (knob.set)(&mut candidate, value);
            assert_eq!((knob.get)(candidate), value);
            if candidate.validate().is_ok() {
                assert_eq!(
                    TuningProfile::from_source(&candidate.render("")),
                    Ok(candidate)
                );
            }
        }
        let objective = CoordinateSearch::objective_weights(
            knob,
            &weights,
            &DivisionGrid::default(),
            &reference,
        );
        for (index, radix) in PARSING_RADICES.into_iter().enumerate() {
            let active = if knob.name == "RADIX_PARSE_LEAF_CHUNKS" {
                true
            } else if knob.name.contains("DECIMAL") {
                radix == 10
            } else if knob.name.contains("SMALL") {
                radix < 10
            } else {
                radix > 10
            };
            let start = index.checked_mul(widths.len()).expect("small catalog");
            let end = start.checked_add(widths.len()).expect("small catalog");
            assert!(
                objective
                    .get(start..end)
                    .expect("radix group")
                    .iter()
                    .all(|&weight| (weight != 0) == active)
            );
        }
    }
}

#[test]
fn every_gcd_policy_candidate_is_valid_and_round_trips() {
    let defaults = TuningProfile::for_target(ARCH, &usize::BITS.to_string());
    for knob in &GCD_KNOBS {
        assert!(matches!(knob.domain, ScoreDomain::Gcd));
        for &candidate in knob.candidates {
            let mut profile = defaults;
            (knob.set)(&mut profile, candidate);
            assert_eq!((knob.get)(profile), candidate);
            if profile.validate().is_err() {
                assert!(matches!(
                    knob.name,
                    "HGCD_BLOCK_THRESHOLD" | "HGCD_CROSSOVER_THRESHOLD"
                ));
                assert!(profile.hgcd_block_threshold > profile.hgcd_crossover);
                continue;
            }
            let parsed = TuningProfile::from_source(&profile.render("// GCD policy test"))
                .expect("candidate profile round trips");
            assert_eq!(parsed, profile);
        }
    }
}

#[test]
fn division_coordinates_are_independent_and_all_valid_candidates_round_trip() {
    let baseline = TuningProfile::portable();
    for knob in DIVISION_KNOBS {
        assert_eq!(knob.domain, ScoreDomain::ProductionDivision);
        for &candidate in knob.candidates {
            let mut profile = baseline;
            (knob.set)(&mut profile, candidate);
            assert_eq!((knob.get)(profile), candidate);
            if profile.validate().is_err() {
                assert_eq!(knob.name, "NEWTON_QUOTIENT_THRESHOLD");
                assert!(profile.newton_quotient <= profile.burnikel_quotient);
                continue;
            }
            assert_eq!(TuningProfile::from_source(&profile.render("")), Ok(profile));
        }
    }
}

#[test]
fn final_division_grid_covers_reference_and_selected_policy_boundaries() {
    let reference = TuningProfile::portable();
    let selected = TuningProfile {
        burnikel_quotient: 251,
        burnikel_long_quotient: 249,
        newton_quotient: 3_071,
        newton_raphson: 511,
        division_stack_limbs: 193,
        division_basecase_quotient_max_limbs: 7,
        ..reference
    };
    assert_eq!(selected.validate(), Ok(()));
    let final_grid = CompiledTuner::division_grid(&[reference, selected], false);
    let search_grid = CompiledTuner::division_grid(&[selected], true);
    for knob in DIVISION_KNOBS {
        let widths = |grid: &DivisionGrid| match knob.name {
            "DIVISION_SINGLE_NORMALIZED_PREINVERSE"
            | "DIVISION_SINGLE_UNNORMALIZED_PREINVERSE"
            | "DIVISION_BASECASE_QUOTIENT_MAX_LIMBS" => grid.quotient_widths.clone(),
            "DIVISION_SMALL_QUOTIENT_MAX" => grid.scalar_quotients.clone(),
            "NEWTON_SMALL_QUOTIENT_BLOCK_RATIO" | "DIVISION_TRUNCATION_RATIO" => {
                grid.block_ratios.clone()
            }
            _ => grid.divisor_widths.clone(),
        };
        for profile in [reference, selected] {
            let value = (knob.get)(profile);
            if value != 0 {
                for width in [value.saturating_sub(1).max(1), value, value + 1] {
                    assert!(
                        widths(&final_grid).contains(&width),
                        "{}: {width}",
                        knob.name
                    );
                }
            }
        }
        for &candidate in knob.candidates {
            if candidate != 0 {
                assert!(widths(&search_grid).contains(&candidate));
            }
        }
    }
    for widths in [
        &final_grid.divisor_widths,
        &final_grid.quotient_widths,
        &final_grid.scalar_quotients,
        &final_grid.block_ratios,
    ] {
        assert!(!widths.contains(&0));
        assert!(
            widths
                .windows(2)
                .all(|pair| matches!(pair, [lower, upper] if lower < upper))
        );
    }
}

#[test]
fn division_policy_objectives_cover_real_cells_and_preserve_all_guards() {
    let grid = DivisionGrid::default();
    let guards = DivisionWorker::cell_weights(&DivisionWorker::cases(&grid));
    for knob in DIVISION_KNOBS {
        let objective =
            CoordinateSearch::objective_weights(&knob, &guards, &grid, &TuningProfile::portable());
        assert_eq!(objective.len(), guards.len());
        assert!(objective.iter().any(|&weight| weight > 0), "{}", knob.name);
        assert!(
            objective
                .iter()
                .zip(&guards)
                .all(|(&weight, &guard)| weight == 0 || weight == guard)
        );
    }
    let single = DIVISION_KNOBS
        .iter()
        .find(|knob| knob.name == "DIVISION_SINGLE_NORMALIZED_PREINVERSE")
        .expect("single-limb admission policy");
    let objective =
        CoordinateSearch::objective_weights(single, &guards, &grid, &TuningProfile::portable());
    let baseline = vec![SCORE_SCALE; guards.len()];
    let candidate: Vec<_> = objective
        .iter()
        .map(|&weight| if weight == 0 { SCORE_SCALE } else { 900_000 })
        .collect();
    assert_eq!(
        CandidateHarness::relative_score(&candidate, &baseline, &objective),
        900_000
    );
    assert!(
        CandidateHarness::relative_score(&candidate, &baseline, &guards) > 992_500,
        "unrelated multi-limb division hid a 10% scalar improvement"
    );
    assert!(Validation::cells_non_regress(
        &candidate,
        &baseline,
        guards.len()
    ));
}

#[test]
fn shortlist_keeps_three_candidates_when_the_coarse_screen_dislikes_all() {
    let mut screened = [
        (4, 1_300_000),
        (8, 1_100_000),
        (12, 1_200_000),
        (16, 1_400_000),
    ];

    assert_eq!(
        CoordinateSearch::shortlist_candidates(&mut screened, 20_000),
        [8, 12, 4]
    );
}

#[test]
fn shortlist_keeps_every_candidate_close_to_the_coarse_winner() {
    let mut screened = [
        (4, 1_000_000),
        (8, 1_010_000),
        (12, 1_020_000),
        (16, 1_035_000),
        (20, 1_050_000),
    ];

    assert_eq!(
        CoordinateSearch::shortlist_candidates(&mut screened, 20_000),
        [4, 8, 12, 16]
    );
}

#[test]
fn cache_block_candidates_stay_sorted_deduplicated_and_bounded() {
    let candidates = CompiledTuner::cache_block_candidates();
    assert!(candidates.len() >= 2);
    assert!(
        candidates
            .windows(2)
            .all(|pair| matches!(pair, [lower, upper] if lower < upper))
    );
    assert!(
        candidates
            .iter()
            .all(|&bytes| (16 * 1024..=4 * 1024 * 1024).contains(&bytes))
    );
}

#[test]
fn each_compiled_coordinate_changes_only_its_named_profile_constant() {
    let baseline = TuningProfile::portable();
    let original = baseline.render("");
    let serial_knobs = TOOM_KNOBS
        .iter()
        .chain(&SSA_KNOBS)
        .chain(&DIVISION_KNOBS)
        .chain(&GCD_KNOBS)
        .chain(PARSING_KNOBS)
        .chain(&PRODUCT_KNOBS)
        .chain(&TRANSFORM_SHAPE_KNOBS);
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    let knobs = serial_knobs.chain(&PARALLEL_KNOBS);
    #[cfg(not(all(feature = "rayon", not(target_pointer_width = "16"))))]
    let knobs = serial_knobs;
    for knob in knobs {
        let value = *knob
            .candidates
            .iter()
            .find(|&&value| value != (knob.get)(baseline))
            .expect("at least one alternative");
        let mut profile = baseline;
        (knob.set)(&mut profile, value);
        let rendered = profile.render("");
        let changed: Vec<_> = original
            .lines()
            .zip(rendered.lines())
            .filter_map(|(before, after)| (before != after).then_some(after))
            .collect();
        assert_eq!(changed.len(), 1, "{}", knob.name);
        assert!(
            changed
                .first()
                .expect("one changed field")
                .starts_with(&format!("pub const {}:", knob.name))
        );
        assert_eq!((knob.get)(profile), value);
    }
}

#[test]
fn local_refinement_checks_neighbours_and_half_intervals_within_search_bounds() {
    let knob = DIVISION_KNOBS
        .iter()
        .find(|knob| knob.name == "BURNIKEL_QUOTIENT_THRESHOLD")
        .expect("quotient coordinate");
    let refined = CoordinateSearch::refine_candidates(knob, 64, &[64]);
    for value in [56, 63, 65, 80] {
        assert!(refined.contains(&value));
    }
    assert_eq!(
        CoordinateSearch::refine_candidates(knob, 64, &[0]),
        Vec::<usize>::new()
    );
    let values = CoordinateSearch::pair_values(knob, 64);
    assert_eq!(values, [48, 63, 64, 65, 96]);
    assert!(
        CoordinateSearch::refine_candidates(knob, 64, &[8, 256])
            .iter()
            .all(|&value| (8..=256).contains(&value))
    );
}

#[test]
fn boundary_expansion_keeps_one_budget_and_envelope_across_passes() {
    let knob = DIVISION_KNOBS
        .iter()
        .find(|knob| knob.name == "BURNIKEL_QUOTIENT_THRESHOLD")
        .expect("quotient coordinate");
    let mut profile = TuningProfile {
        burnikel_quotient: 256,
        ..TuningProfile::portable()
    };
    let mut grid = CoordinateGrid::new(knob, &profile);
    assert!(grid.expand(knob, &profile));
    assert!(grid.values.contains(&512));
    profile.burnikel_quotient = 512;
    assert!(grid.expand(knob, &profile));
    assert!(grid.values.contains(&1_024));
    profile.burnikel_quotient = 1_024;
    assert!(!grid.expand(knob, &profile));
    assert!(!grid.values.contains(&2_048));
    assert!(
        grid.values
            .windows(2)
            .all(|pair| matches!(pair, [a, b] if a < b))
    );
}

#[test]
fn expanded_endpoints_and_both_neighbours_are_frozen_before_measurement() {
    let knobs = [
        PARSING_KNOBS.get(1).expect("decimal parsing"),
        PRODUCT_KNOBS
            .iter()
            .find(|knob| knob.name == "MONTGOMERY_CIOS_MAX_LIMBS")
            .expect("CIOS"),
        DIVISION_KNOBS
            .iter()
            .find(|knob| knob.name == "BURNIKEL_QUOTIENT_THRESHOLD")
            .expect("quotient"),
    ];
    for original in knobs {
        let mut profile = TuningProfile::portable();
        let edge = original
            .candidates
            .iter()
            .copied()
            .max()
            .expect("grid edge");
        (original.set)(&mut profile, edge);
        assert_eq!(profile.validate(), Ok(()));
        let mut grid = CoordinateGrid::new(original, &profile);
        assert!(grid.expand(original, &profile));
        let knob = Knob {
            candidates: &grid.values,
            ..*original
        };
        let profiles = CoordinateSearch::grid_profiles(&knob, &profile);
        let expanded = *grid.values.last().expect("expanded endpoint");
        assert!(profiles.iter().any(|&trial| (knob.get)(trial) == expanded));
        let widths = if knob.domain == ScoreDomain::Parsing {
            CompiledTuner::parsing_grid(&profiles, false)
        } else if knob.domain == ScoreDomain::Products {
            CompiledTuner::product_grid(&profiles, false).montgomery
        } else {
            assert_eq!(knob.domain, ScoreDomain::ProductionDivision);
            CompiledTuner::division_grid(&profiles, false).divisor_widths
        };
        for neighbour in [expanded - 1, expanded, expanded + 1] {
            assert!(widths.contains(&neighbour), "{}: {neighbour}", knob.name);
        }
    }
}

#[test]
fn expansion_can_cross_a_lower_grid_edge_without_introducing_a_strategy_switch() {
    let original = PRODUCT_KNOBS
        .iter()
        .find(|knob| knob.name == "MONTGOMERY_CIOS_MAX_LIMBS")
        .expect("Montgomery coordinate");
    let knob = Knob {
        candidates: &[4, 8, 16],
        ..*original
    };
    let mut profile = TuningProfile {
        montgomery_cios_max_limbs: 4,
        ..TuningProfile::portable()
    };
    let mut grid = CoordinateGrid::new(&knob, &profile);
    assert!(grid.expand(&knob, &profile));
    profile.montgomery_cios_max_limbs = 2;
    assert!(grid.expand(&knob, &profile));
    assert!(grid.values.contains(&1));
    assert!(!grid.values.contains(&0));
    let disabled = PRODUCT_KNOBS
        .iter()
        .find(|candidate| candidate.name == "MUL_MOD_BNM1_THRESHOLD")
        .expect("optional cyclic coordinate");
    profile.mul_mod_bnm1 = 0;
    let mut disabled_grid = CoordinateGrid::new(disabled, &profile);
    assert!(!disabled_grid.expand(disabled, &profile));
}

#[test]
fn expansion_preserves_mathematical_bounds_and_profile_ordering() {
    let knob = SSA_KNOBS
        .iter()
        .find(|knob| knob.name == "SSA_SHIFT_BLOCK_WIDTH")
        .expect("blocked shift coordinate");
    let profile = TuningProfile {
        ssa_shift_block_width: 256,
        ..TuningProfile::portable()
    };
    let mut grid = CoordinateGrid::new(knob, &profile);
    assert!(!grid.expand(knob, &profile));
    assert!(!grid.values.contains(&512));
    let leaf = PARSING_KNOBS.first().expect("parsing leaf");
    let parsing_profile = TuningProfile {
        radix_parse_leaf: 64,
        radix_parse_decimal_recursive: 128,
        radix_parse_small_recursive: 128,
        radix_parse_large_recursive: 128,
        ..TuningProfile::portable()
    };
    assert_eq!(parsing_profile.validate(), Ok(()));
    let mut parsing_grid = CoordinateGrid::new(leaf, &parsing_profile);
    assert!(!parsing_grid.expand(leaf, &parsing_profile));
    assert!(!parsing_grid.values.contains(&128));
}

#[test]
fn reserved_validation_cannot_be_cached_or_used_as_a_search_objective() {
    let directory = temp_dir().join(format!("mp-tune-holdout-{}", process_id()));
    let defaults = TuningProfile::portable();
    let mut harness = CandidateHarness::new(&directory.join("scores"), 0).expect("test harness");
    assert!(
        harness
            .score(&defaults, ScoreDomain::Holdout, false)
            .is_none()
    );
    assert!(harness.failed);
    assert_eq!(harness.confirmations, 0);
    let serial_knobs = TOOM_KNOBS
        .iter()
        .chain(&SSA_KNOBS)
        .chain(&DIVISION_KNOBS)
        .chain(&GCD_KNOBS)
        .chain(PARSING_KNOBS)
        .chain(&PRODUCT_KNOBS)
        .chain(&TRANSFORM_SHAPE_KNOBS);
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    let knobs = serial_knobs.chain(&PARALLEL_KNOBS);
    #[cfg(not(all(feature = "rayon", not(target_pointer_width = "16"))))]
    let knobs = serial_knobs;
    for knob in knobs {
        assert_ne!(knob.domain, ScoreDomain::Holdout);
    }
    drop(harness);
    remove_dir_all(directory).expect("remove test artifacts");
}

#[test]
fn joint_quotient_updates_can_cross_a_profile_ordering_constraint() {
    let baseline = TuningProfile::portable();
    let a = DIVISION_KNOBS
        .iter()
        .find(|knob| knob.name == "BURNIKEL_QUOTIENT_THRESHOLD")
        .expect("Burnikel cutoff");
    let b = DIVISION_KNOBS
        .iter()
        .find(|knob| knob.name == "NEWTON_QUOTIENT_THRESHOLD")
        .expect("Newton cutoff");
    let mut profile = baseline;
    (a.set)(&mut profile, baseline.newton_quotient + 1);
    assert!(profile.validate().is_err());
    (b.set)(&mut profile, baseline.newton_quotient + 2);
    assert_eq!(profile.validate(), Ok(()));
}

#[test]
fn scalar_preinversion_plans_execute_only_their_normalization_class() {
    let directory = temp_dir().join(format!("mp-tune-cell-plan-{}", process_id()));
    let defaults = TuningProfile::portable();
    let mut session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 0,
            timing_bucket_ms: 0,
        },
        directory.clone(),
        "test".to_owned(),
        "test".to_owned(),
    )
    .expect("session artifacts can be created");
    let mut cells = Vec::new();
    for case in DivisionWorker::cases(&session.harness.division_grid) {
        for fixture in 0..if case.scalar_quotient.is_some() { 6 } else { 4 } {
            for output in 0..4 {
                cells.push((case.divisor_limbs, fixture, output));
            }
        }
    }
    for (name, normalized) in [
        ("DIVISION_SINGLE_NORMALIZED_PREINVERSE", true),
        ("DIVISION_SINGLE_UNNORMALIZED_PREINVERSE", false),
    ] {
        let knob = *DIVISION_KNOBS
            .iter()
            .find(|knob| knob.name == name)
            .expect("preinverse coordinate");
        let plan = CoordinateSearch::measurement_plan(&mut session, &[knob]);
        assert_ne!(plan.indices, Vec::<usize>::new());
        assert_eq!(session.harness.selected_cells, plan.indices);
        for &index in &plan.indices {
            let &(divisor, fixture, output) = cells.get(index).expect("frozen cell index");
            assert_eq!(divisor, 1);
            assert_eq!(fixture >= 2, normalized);
            assert!(output < 3);
        }
        assert_eq!(plan.indices.len(), plan.weights.len());
    }
    drop(session);
    remove_dir_all(directory).expect("remove test artifacts");
}
