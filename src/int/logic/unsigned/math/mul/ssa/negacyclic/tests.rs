//! Odd-factor products against full-width Fermat reduction.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Negacyclic tests construct complete initialized coefficient spans and exact workspaces before exercising unsafe kernels"
)]
use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;

fn dense_operand(len: usize, mut state: u64) -> Vec<Limb> {
    state ^= 0x9e37_79b9_7f4a_7c15;
    let mut limbs = Vec::with_capacity(len.wrapping_add(1));
    for _ in 0..len {
        state ^= state.wrapping_shl(13);
        state ^= state.wrapping_shr(7);
        state ^= state.wrapping_shl(17);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "the deterministic test stream is intentionally truncated to the target limb width"
        )]
        limbs.push(state as Limb);
    }
    limbs.push(0);
    limbs
}

fn check_factorized_product(modulus_limbs: usize, left_seed: u64, right_seed: u64) {
    let factor = if modulus_limbs.is_multiple_of(5) {
        5
    } else {
        3
    };
    let plan = NegacyclicPlan::for_factor(
        modulus_limbs,
        core::num::NonZeroUsize::new(factor).expect("nonzero factor"),
    )
    .expect("test width has an admitted odd factor");
    let modulus_bits = modulus_limbs.wrapping_mul(LIMB_BITS);
    let original = dense_operand(modulus_limbs, left_seed);
    let right = dense_operand(modulus_limbs, right_seed);
    let mut expected = vec![0; modulus_limbs.wrapping_add(1)];
    let expected_scratch_len = SsaPointwise::fermat_basecase_scratch_len(modulus_bits);
    let mut exact_scratch = vec![0; expected_scratch_len];
    // SAFETY: all operands and scratch buffers have the exact coefficient
    // widths required by the two internal product implementations.
    unsafe {
        SsaPointwise::fermat_basecase_mul_into(
            &mut expected,
            &original,
            &right,
            modulus_bits,
            &mut exact_scratch,
        );
    }

    let mut factor_scratch = vec![Limb::MAX; plan.scratch_len + 2];
    let mut result = vec![Limb::MAX; modulus_limbs.wrapping_add(1)];
    for dirty in [Limb::MAX, 0] {
        let mut left = original.clone();
        factor_scratch[1..=plan.scratch_len].fill(dirty);
        // SAFETY: complete disjoint canonical operands have zero guards;
        // the borrowed initialized arena has exactly the plan's scratch size.
        unsafe {
            plan.mul_assign_left(
                &mut left,
                &right,
                &mut result,
                &mut factor_scratch[1..=plan.scratch_len],
            );
        }
        assert_eq!(left, expected);
    }
    assert_eq!(factor_scratch[0], Limb::MAX, "leading scratch canary");
    assert_eq!(
        factor_scratch[plan.scratch_len + 1],
        Limb::MAX,
        "trailing scratch canary"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 16 }))]

    #[test]
    fn reused_odd_factor_arenas_cover_dense_products_and_block_boundaries(
        left_seed in any::<u64>(),
        right_seed in any::<u64>(),
        block in if cfg!(miri) { (1_usize..=4).boxed() } else { prop_oneof![1_usize..=12, 63_usize..=65, 95_usize..=97].boxed() },
        factor in prop_oneof![Just(3_usize), Just(5_usize)],
    ) {
        check_factorized_product(block * factor, left_seed, right_seed);
    }
}

#[test]
fn factorized_product_handles_sparse_boundaries() {
    // Miri checks the same factor/block/guard transitions at short data widths;
    // native execution additionally crosses the recursive multiplication tiers.
    let widths = if cfg!(miri) {
        [(6, 3), (10, 5)]
    } else {
        [(288, 3), (320, 5)]
    };
    for (width, factor) in widths {
        let plan = NegacyclicPlan::for_factor(
            width,
            core::num::NonZeroUsize::new(factor).expect("nonzero factor"),
        )
        .expect("odd factor width");
        let block = width.div_euclid(factor);
        let bits = width * LIMB_BITS;
        let mut scratch = vec![0; plan.scratch_len];
        let mut result = vec![0; width + 1];
        let mut exact_scratch = vec![0; SsaPointwise::fermat_basecase_scratch_len(bits)];
        for i in 0..factor {
            for j in 0..factor {
                for folded_zero in [false, true] {
                    let mut left = vec![0; width + 1];
                    let mut right = vec![0; width + 1];
                    // SAFETY: both single-limb stores land on exact block
                    // boundaries below width + 1: i and j stay below factor.
                    unsafe {
                        *left.get_unchecked_mut(i.wrapping_mul(block)) = 1;
                        *right.get_unchecked_mut(j.wrapping_mul(block)) = 1;
                    }
                    if folded_zero {
                        // SAFETY: block powers of the outer width stay below
                        // width + 1 for the same block partition.
                        unsafe {
                            *left.get_unchecked_mut(0) = 1;
                            *left.get_unchecked_mut(block) = 1;
                        }
                    }
                    let mut expected = vec![0; width + 1];
                    let mut second = left.clone();
                    // SAFETY: canonical ordinary outer residues and complete
                    // disjoint workspaces meet both fixed-width contracts.
                    unsafe {
                        SsaPointwise::fermat_basecase_mul_into(
                            &mut expected,
                            &left,
                            &right,
                            bits,
                            &mut exact_scratch,
                        );
                        plan.mul_assign_left(&mut left, &right, &mut result, &mut scratch);
                        plan.mul_assign_left(&mut second, &right, &mut result, &mut scratch);
                    }
                    assert_eq!(left, expected, "width {width}, powers {i}, {j}");
                    assert_eq!(second, expected, "shared operand remains reusable");
                }
            }
        }
    }
}

#[test]
fn quotient_modulus_comparison_progresses_across_exact_blocks() {
    for factor in [3, 5] {
        for block_len in [1, 2, 3, 8] {
            let plan = NegacyclicPlan::for_factor(
                block_len * factor,
                NonZeroUsize::new(factor).expect("positive test factor"),
            )
            .expect("small exact quotient geometry");
            let mut quotient = vec![0; plan.quotient_coeff_len];
            quotient[0] = 1;
            for block in (1..factor - 1).step_by(2) {
                quotient[block * block_len..(block + 1) * block_len].fill(Limb::MAX);
            }
            assert_eq!(
                // SAFETY: the constructed reference contains every exact data
                // block and one initialized guard reported by this admitted plan.
                unsafe { plan.compare_with_quotient_modulus(&quotient) },
                core::cmp::Ordering::Equal,
                "exact Q for factor {factor}, block width {block_len}"
            );
            for index in 0..quotient.len() {
                let mut changed = quotient.clone();
                for ordering in [core::cmp::Ordering::Less, core::cmp::Ordering::Greater] {
                    let digit = quotient[index];
                    match ordering {
                        core::cmp::Ordering::Less if digit > 0 => changed[index] = digit - 1,
                        core::cmp::Ordering::Greater if digit < Limb::MAX => {
                            changed[index] = digit + 1;
                        }
                        core::cmp::Ordering::Less
                        | core::cmp::Ordering::Equal
                        | core::cmp::Ordering::Greater => continue,
                    }
                    assert_eq!(
                        // SAFETY: modifying one digit preserves the complete
                        // initialized coefficient, including its retained guard.
                        unsafe { plan.compare_with_quotient_modulus(&changed) },
                        ordering,
                        "factor {factor}, block width {block_len}, digit {index}"
                    );
                }
            }
        }
    }
}
