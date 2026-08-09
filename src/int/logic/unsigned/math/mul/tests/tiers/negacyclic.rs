//! Property tests for negacyclic multiplication.

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
        #[allow(
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
    let plan = NegacyclicPlan::new(modulus_limbs).expect("test width has an odd factor");
    let modulus_bits = modulus_limbs.wrapping_mul(LIMB_BITS);
    let mut left = dense_operand(modulus_limbs, left_seed);
    let mut right = dense_operand(modulus_limbs, right_seed);
    let mut expected = vec![0; modulus_limbs.wrapping_add(1)];
    let expected_scratch_len = SsaPointwise::fermat_basecase_scratch_len(modulus_bits);
    let mut exact_scratch = vec![0; expected_scratch_len];
    // SAFETY: all operands and scratch buffers have the exact coefficient
    // widths required by the two internal product implementations.
    unsafe {
        SsaPointwise::fermat_basecase_mul_into(
            &mut expected,
            &left,
            &right,
            modulus_bits,
            &mut exact_scratch,
        );
    }

    let mut factor_scratch = vec![0; plan.scratch_len()];
    // SAFETY: the generated residues are canonical with zero guard limbs,
    // and the plan sized its own scratch buffer.
    unsafe {
        plan.mul_assign_left(&mut left, &mut right, &mut factor_scratch);
    }
    assert_eq!(left, expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_odd_factor_product_matches_full_product(
        left_seed in any::<u64>(),
        right_seed in any::<u64>(),
        use_factor_five in any::<bool>(),
    ) {
        let modulus_limbs = if use_factor_five { 320 } else { 288 };
        check_factorized_product(modulus_limbs, left_seed, right_seed);
    }
}

#[test]
fn factorized_product_handles_sparse_boundaries() {
    check_factorized_product(288, 0, u64::MAX);
    check_factorized_product(320, u64::MAX, 0);
}
