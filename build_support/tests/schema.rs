//! Scalar policy ranges, recursive geometry, and coupled dispatch constraints.

use super::super::TuningProfile;

type Policy = (&'static str, fn(&mut TuningProfile, usize), usize, usize);

macro_rules! policies {
    ($($field:ident => ($minimum:expr, $maximum:expr)),+ $(,)?) => {
        [$((stringify!($field), |profile: &mut TuningProfile, value| profile.$field = value,
            $minimum, $maximum)),+]
    };
}

#[test]
fn scalar_policy_ranges_accept_boundaries_and_reject_invalid_inputs() {
    let baseline = TuningProfile::portable();
    let finite_maximum = usize::MAX - 2;
    let recursive_minimum = baseline.radix_parse_leaf.checked_add(1).expect("leaf<MAX");
    let leaf_maximum = baseline
        .radix_parse_decimal_recursive
        .min(baseline.radix_parse_small_recursive)
        .min(baseline.radix_parse_large_recursive)
        .checked_sub(1)
        .expect("portable recursive widths are positive");
    let policies: &[Policy] = &policies![
        radix_format_decimal_recursive => (1, usize::MAX),
        radix_format_small_recursive => (1, usize::MAX),
        radix_format_large_recursive => (1, usize::MAX),
        radix_parse_decimal_recursive => (recursive_minimum, finite_maximum),
        radix_parse_small_recursive => (recursive_minimum, finite_maximum),
        radix_parse_large_recursive => (recursive_minimum, finite_maximum),
        radix_parse_leaf => (1, leaf_maximum),
        hgcd_crossover => (baseline.hgcd_block_threshold, finite_maximum),
        extended_hgcd_crossover => (1, finite_maximum),
        extended_gcd_wide_threshold => (1, finite_maximum),
        extended_gcd_cofactor_batch_min => (1, finite_maximum),
        extended_gcd_cofactor_batch_ratio => (1, finite_maximum),
        hgcd_block_threshold => (1, baseline.hgcd_crossover),
        lehmer_fused_update_max => (1, finite_maximum),
        wide_lehmer_threshold => (1, finite_maximum),
        lehmer_branchless_threshold => (1, finite_maximum),
        binary_euclid_division_shift => (1, 64),
        low_product_recursive => (4, finite_maximum),
        low_product_full => (0, finite_maximum),
        lopsided_transform_block_ratio => (1, finite_maximum),
        toom85_paired_reconstruction_min_limbs => (1, finite_maximum),
        toom8_full_guard_product_min_split_limbs => (1, finite_maximum),
        burnikel_ziegler => (1, baseline.newton_raphson),
        newton_raphson => (baseline.burnikel_ziegler, finite_maximum),
        burnikel_quotient => (1, baseline.newton_quotient),
        burnikel_long_quotient => (1, baseline.newton_quotient),
        newton_quotient => (baseline.burnikel_quotient.max(baseline.burnikel_long_quotient), finite_maximum),
        division_small_quotient_max => (2, usize::from(u16::MAX)),
        newton_reciprocal_basecase => (1, finite_maximum),
        approximate_division_block => (4, finite_maximum),
        newton_small_quotient_block_ratio => (1, finite_maximum),
        division_truncation_ratio => (1, finite_maximum),
        mul_mod_bnm1 => (0, finite_maximum),
        division_divisible => (1, finite_maximum),
        division_stack_limbs => (4, 512),
        division_single_normalized_preinverse => (2, finite_maximum),
        division_single_unnormalized_preinverse => (2, finite_maximum),
        division_basecase_quotient_max_limbs => (1, finite_maximum),
        montgomery_pow_mod => (1, finite_maximum),
        montgomery_cios_max_limbs => (1, finite_maximum),
        transform_min_smaller_limbs => (1, finite_maximum),
        transform_max_operand_ratio => (1, finite_maximum),
        ssa_base_modulus_bits => (1, finite_maximum),
        ssa_bnm1_basecase_limbs => (1, finite_maximum),
        ssa_negacyclic_factor3 => (1, finite_maximum),
        ssa_negacyclic_factor5 => (1, finite_maximum),
        ssa_coefficient_visit_overhead => (1, finite_maximum),
        ssa_basecase_cost_weight_16ths => (1, finite_maximum),
        ssa_nested_cost_penalty_16ths => (1, finite_maximum),
        ssa_direct_shift_max_limbs => (0, finite_maximum),
        ssa_shift_scalar_threshold => (1, finite_maximum),
        ssa_direct_fermat_parallel_threshold => (0, finite_maximum),
        ssa_direct_fermat_parallel_min_workers => (1, finite_maximum),
        ssa_parallel_min_limb_work => (1, finite_maximum),
        cache_block_bytes => (1, finite_maximum),
    ];
    for &(name, set, minimum, maximum) in policies {
        let candidates = [
            0,
            1,
            2,
            3,
            4,
            16,
            64,
            65,
            256,
            257,
            512,
            513,
            minimum.saturating_sub(1),
            minimum,
            minimum.saturating_add(1),
            maximum.saturating_sub(1),
            maximum,
            maximum.saturating_add(1),
            usize::MAX - 1,
            usize::MAX,
        ];
        for value in candidates {
            let mut profile = baseline;
            set(&mut profile, value);
            assert_eq!(
                profile.validate().is_ok(),
                (minimum..=maximum).contains(&value),
                "{name} = {value}, accepted range {minimum}..={maximum}"
            );
        }
    }
}

#[test]
fn block_geometry_requires_equal_halves_or_bounded_power_of_two_storage() {
    for block in (0..=513).chain([usize::MAX - 2, usize::MAX - 1, usize::MAX]) {
        let division_profile = TuningProfile {
            burnikel_ziegler_block: block,
            ..TuningProfile::portable()
        };
        assert_eq!(
            division_profile.validate().is_ok(),
            block != 0 && block < usize::MAX - 1 && block.is_multiple_of(2),
            "division block {block}"
        );
        let shift_profile = TuningProfile {
            ssa_shift_block_width: block,
            ..TuningProfile::portable()
        };
        assert_eq!(
            shift_profile.validate().is_ok(),
            block.is_power_of_two() && block <= 256,
            "shift block {block}"
        );
    }
}

#[test]
fn multiplication_and_squaring_ladders_preserve_order() {
    let baseline = TuningProfile::portable();
    let setters: &[fn(&mut TuningProfile, usize)] = &[
        |profile, value| profile.karatsuba = value,
        |profile, value| profile.toom_cook_3 = value,
        |profile, value| profile.toom_cook_4 = value,
        |profile, value| profile.toom_cook_6 = value,
        |profile, value| profile.toom_cook_85 = value,
        |profile, value| profile.sqr_karatsuba = value,
        |profile, value| profile.sqr_toom_cook_3 = value,
        |profile, value| profile.sqr_toom_cook_4 = value,
        |profile, value| profile.sqr_toom_cook_6 = value,
        |profile, value| profile.sqr_toom_cook_85 = value,
    ];
    for (index, set) in setters.iter().enumerate() {
        for value in [
            0,
            1,
            28,
            191,
            259,
            278,
            313,
            512,
            2_049,
            3_073,
            usize::MAX - 1,
            usize::MAX,
        ] {
            let mut profile = baseline;
            set(&mut profile, value);
            let chain = if index < 5 {
                [
                    profile.karatsuba,
                    profile.toom_cook_3,
                    profile.toom_cook_4,
                    profile.toom_cook_6,
                    profile.toom_cook_85,
                ]
            } else {
                [
                    profile.sqr_karatsuba,
                    profile.sqr_toom_cook_3,
                    profile.sqr_toom_cook_4,
                    profile.sqr_toom_cook_6,
                    profile.sqr_toom_cook_85,
                ]
            };
            let transform = if index < 5 {
                profile.ssa
            } else {
                profile.sqr_ssa
            };
            let expected = chain.iter().all(|&entry| entry != 0 && entry != usize::MAX)
                && chain
                    .windows(2)
                    .all(|pair| matches!(pair, [low, high] if low <= high))
                && transform
                    > chain
                        .iter()
                        .copied()
                        .filter(|&entry| entry < usize::MAX - 1)
                        .max()
                        .unwrap_or(0)
                && (index >= 5 || profile.balanced_toom8 > profile.karatsuba);
            assert_eq!(
                profile.validate().is_ok(),
                expected,
                "ladder field {index} = {value}"
            );
        }
    }
}

#[test]
fn optional_multiplication_tiers_follow_the_active_ladder() {
    let baseline = TuningProfile::portable();
    for balanced in [
        0,
        baseline.karatsuba,
        baseline.karatsuba + 1,
        baseline.ssa - 1,
        baseline.ssa,
        usize::MAX - 1,
        usize::MAX,
    ] {
        let profile = TuningProfile {
            balanced_toom8: balanced,
            ..baseline
        };
        assert_eq!(
            profile.validate().is_ok(),
            balanced == 0 || (balanced > baseline.karatsuba && balanced < baseline.ssa)
        );
    }
    for transform in [
        0,
        baseline.toom_cook_85,
        baseline.toom_cook_85 + 1,
        usize::MAX - 2,
        usize::MAX - 1,
        usize::MAX,
    ] {
        for square in [false, true] {
            let mut profile = baseline;
            if square {
                profile.sqr_ssa = transform;
            } else {
                profile.ssa = transform;
            }
            assert_eq!(
                profile.validate().is_ok(),
                transform == 0 || (transform > baseline.toom_cook_85 && transform < usize::MAX - 1)
            );
        }
    }
    let disabled_tail = TuningProfile {
        toom_cook_6: usize::MAX - 1,
        toom_cook_85: usize::MAX - 1,
        sqr_toom_cook_6: usize::MAX - 1,
        sqr_toom_cook_85: usize::MAX - 1,
        ..baseline
    };
    assert_eq!(
        disabled_tail.validate(),
        Ok(()),
        "disabled tail allows transforms after active tiers"
    );
    let moved_transform = TuningProfile {
        toom_cook_85: 4_096,
        ssa: 5_000,
        ..baseline
    };
    assert_eq!(
        moved_transform.validate(),
        Ok(()),
        "conventional and transform entries may move together"
    );
}
