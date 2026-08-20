//! Comprehensive verification tests for 50-bit floating-point Harvey NTT multiplication and squaring.

#![allow(
    unsafe_code,
    reason = "Test verification of internal unsafe conversion kernels"
)]

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::{
    super::Schoolbook, FLOAT_PINV_1, FLOAT_PINV_2, FLOAT_PINV_3, FLOAT_PRIME_1, FLOAT_PRIME_1_INT,
    FLOAT_PRIME_2, FLOAT_PRIME_2_INT, FLOAT_PRIME_3, FLOAT_PRIME_3_INT, FLOAT_ROOT_1, FLOAT_ROOT_2,
    FLOAT_ROOT_3, Limb, Ntt,
};
use crate::parallel::{DefaultExecutor, SequentialExecutor};

#[test]
fn test_float50_constants() {
    assert_eq!(FLOAT_PRIME_1_INT, 1_108_307_720_798_209);
    assert_eq!(FLOAT_PRIME_2_INT, 1_086_317_488_242_689);
    assert_eq!(FLOAT_PRIME_3_INT, 910_395_627_798_529);

    assert_eq!(FLOAT_PRIME_1_INT % (1 << 30), 1);
    assert_eq!(FLOAT_PRIME_2_INT % (1 << 30), 1);
    assert_eq!(FLOAT_PRIME_3_INT % (1 << 30), 1);
}

#[test]
fn test_float50_twiddles_generation() {
    for (prime_int, prime, pinv, root) in [
        (FLOAT_PRIME_1_INT, FLOAT_PRIME_1, FLOAT_PINV_1, FLOAT_ROOT_1),
        (FLOAT_PRIME_2_INT, FLOAT_PRIME_2, FLOAT_PINV_2, FLOAT_ROOT_2),
        (FLOAT_PRIME_3_INT, FLOAT_PRIME_3, FLOAT_PINV_3, FLOAT_ROOT_3),
    ] {
        let mut tw = [0.0_f64; 64];
        Ntt::generate_stage_twiddles(&mut tw, 64, root, prime_int, prime, pinv, false);
        let first_tw = tw.first().copied().unwrap_or(0.0);
        assert!((first_tw - 1.0_f64).abs() < f64::EPSILON);

        let mut inv_tw = [0.0_f64; 64];
        Ntt::generate_stage_twiddles(&mut inv_tw, 64, root, prime_int, prime, pinv, true);
        let first_inv_tw = inv_tw.first().copied().unwrap_or(0.0);
        assert!((first_inv_tw - 1.0_f64).abs() < f64::EPSILON);
    }
}

#[test]
fn test_float50_mulmod_exactness() {
    let a = 123_456_789_012_345.0_f64;
    let b = 987_654_321_098_765.0_f64;

    for (p, pinv, p_int) in [
        (FLOAT_PRIME_1, FLOAT_PINV_1, FLOAT_PRIME_1_INT),
        (FLOAT_PRIME_2, FLOAT_PINV_2, FLOAT_PRIME_2_INT),
        (FLOAT_PRIME_3, FLOAT_PINV_3, FLOAT_PRIME_3_INT),
    ] {
        let res = Ntt::mulmod(a, b, p, pinv);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::arithmetic_side_effects,
            reason = "Test verification of 50-bit exact values"
        )]
        let (a_int, b_int, p_u128) = (a as u128, b as u128, u128::from(p_int));
        #[allow(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            clippy::arithmetic_side_effects,
            reason = "Test verification modulo fits in 50-bit mantissa"
        )]
        let expected = ((a_int.wrapping_mul(b_int)) % p_u128) as f64;
        let diff = (res - expected).abs();
        assert!(
            diff < 1e-3 || (diff - p).abs() < 1e-3,
            "mulmod error: got {res}, expected {expected} mod {p}"
        );
    }
}

#[test]
fn test_float50_ntt_roundtrip() {
    let limbs = [0x1234_5678_usize, 0x9ABC_DEF0_usize];
    let mut digits = [0.0_f64; 16];
    // SAFETY: digits capacity is 16.
    let count = unsafe { Ntt::limbs_to_digits_50_into(&mut digits, &limbs) };
    assert!(count <= 4);

    let primes_and_roots = [
        (FLOAT_PRIME_1_INT, FLOAT_PRIME_1, FLOAT_PINV_1, FLOAT_ROOT_1),
        (FLOAT_PRIME_2_INT, FLOAT_PRIME_2, FLOAT_PINV_2, FLOAT_ROOT_2),
        (FLOAT_PRIME_3_INT, FLOAT_PRIME_3, FLOAT_PINV_3, FLOAT_ROOT_3),
    ];

    for (p_int, p, pinv, root) in primes_and_roots {
        let mut buffer = digits;
        let mut tw_fwd = [0.0_f64; 16];
        let mut tw_inv = [0.0_f64; 16];

        Ntt::generate_stage_twiddles(&mut tw_fwd, 16, root, p_int, p, pinv, false);
        Ntt::generate_stage_twiddles(&mut tw_inv, 16, root, p_int, p, pinv, true);

        Ntt::forward_transform(&mut buffer, &tw_fwd, 16, p, pinv, &SequentialExecutor);
        Ntt::inverse_transform(&mut buffer, &tw_inv, 16, p, pinv, &SequentialExecutor);

        let inv16 = Ntt::pow_mod_float(16.0, p_int.wrapping_sub(2), p, pinv);
        for slot in &mut buffer {
            *slot = Ntt::mulmod(*slot, inv16, p, pinv);
        }

        for (i, (&buf_elem, &digit_elem)) in buffer.iter().zip(digits.iter()).enumerate() {
            let mut diff = (buf_elem - digit_elem).abs();
            if (diff - p).abs() < 1e-3 {
                diff = 0.0;
            }
            assert!(
                diff < 1e-3,
                "Mismatch at {i}: got {buf_elem}, expected {digit_elem}"
            );
        }
    }
}

#[test]
fn test_float50_try_mul_matches_schoolbook() {
    let executor = SequentialExecutor;

    for (len_a, len_b) in [
        (8, 8),
        (16, 16),
        (20, 20),
        (32, 32),
        (48, 48),
        (64, 64),
        (128, 128),
    ] {
        let a: Vec<Limb> = (0..len_a)
            .map(|i: usize| i.wrapping_mul(0x517C_C1B7_2722_0A95_usize).wrapping_add(3))
            .collect();
        let b: Vec<Limb> = (0..len_b)
            .map(|i: usize| i.wrapping_mul(0x9E37_79B9_7F4A_7C15_usize).wrapping_add(5))
            .collect();

        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0_usize; result_len];
        let mut actual = vec![0_usize; result_len];

        Schoolbook::mul(&mut expected, &a, &b);
        let ok = Ntt::try_mul_with_executor(&mut actual, &a, &b, None, &executor);
        assert!(
            ok,
            "try_mul_with_executor failed for lengths ({len_a}, {len_b})"
        );
        assert_eq!(
            actual, expected,
            "Float50 NTT product {actual:?} differs from schoolbook {expected:?} for lengths ({len_a}, {len_b})"
        );
    }
}

#[test]
fn test_float50_squaring() {
    let executor = SequentialExecutor;

    for len in [8, 16, 20, 32, 48, 64, 128] {
        let a: Vec<Limb> = (0..len)
            .map(|i: usize| i.wrapping_mul(0x517C_C1B7_2722_0A95_usize).wrapping_add(9))
            .collect();

        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0_usize; result_len];
        let mut actual = vec![0_usize; result_len];

        Schoolbook::sqr(&mut expected, &a);
        let ok = Ntt::try_sqr_with_executor(&mut actual, &a, None, &executor);
        assert!(ok, "try_sqr_with_executor failed for length {len}");
        assert_eq!(
            actual, expected,
            "Float50 NTT square differs from schoolbook for length {len}"
        );
    }
}

#[test]
fn test_float50_parallel_executor() {
    let executor = DefaultExecutor::default();

    for (len_a, len_b) in [(16, 16), (32, 32), (64, 64), (128, 128)] {
        let a: Vec<Limb> = (0..len_a)
            .map(|i: usize| i.wrapping_mul(0x517C_C1B7_2722_0A95_usize).wrapping_add(7))
            .collect();
        let b: Vec<Limb> = (0..len_b)
            .map(|i: usize| i.wrapping_mul(0x9E37_79B9_7F4A_7C15_usize).wrapping_add(11))
            .collect();

        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0_usize; result_len];
        let mut actual = vec![0_usize; result_len];

        Schoolbook::mul(&mut expected, &a, &b);
        let ok = Ntt::try_mul_with_executor(&mut actual, &a, &b, None, &executor);
        assert!(
            ok,
            "parallel try_mul_with_executor failed for lengths ({len_a}, {len_b})"
        );
        assert_eq!(
            actual, expected,
            "parallel Float50 NTT product differs from schoolbook for lengths ({len_a}, {len_b})"
        );
    }
}

fn operand_strategy() -> impl Strategy<Value = Vec<Limb>> {
    prop::collection::vec(any::<usize>(), 8..=64)
}

fn operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    (operand_strategy(), operand_strategy())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_ntt_mul_matches_schoolbook((a, b) in operands()) {
        let executor = SequentialExecutor;
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0_usize; result_len];
        let mut actual = vec![0_usize; result_len];

        Schoolbook::mul(&mut expected, &a, &b);
        let ok = Ntt::try_mul_with_executor(&mut actual, &a, &b, None, &executor);
        prop_assert!(ok, "NTT multiplication failed");
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_mul_with_scratch_matches_executor((a, b) in operands()) {
        let executor = SequentialExecutor;
        let result_len = a.len().wrapping_add(b.len());
        let mut actual_allocating = vec![0_usize; result_len];
        let mut actual_scratch = vec![0_usize; result_len];

        let cap_a = Ntt::digit_capacity(a.len(), 50).expect("capacity fits");
        let cap_b = Ntt::digit_capacity(b.len(), 50).expect("capacity fits");
        let transform_len = Ntt::transform_capacity(cap_a, cap_b).expect("transform capacity fits");
        let mut scratch = vec![0.0_f64; Ntt::scratch_len(transform_len)];

        let ok1 = Ntt::try_mul_with_executor(&mut actual_allocating, &a, &b, None, &executor);
        let ok2 = Ntt::try_mul_with_scratch(&mut actual_scratch, &a, &b, &mut scratch, &executor);

        prop_assert!(ok1 && ok2);
        prop_assert_eq!(actual_allocating, actual_scratch);
    }

    #[test]
    fn prop_ntt_sqr_matches_schoolbook(a in operand_strategy()) {
        let executor = SequentialExecutor;
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0_usize; result_len];
        let mut actual = vec![0_usize; result_len];

        Schoolbook::sqr(&mut expected, &a);
        let ok = Ntt::try_sqr_with_executor(&mut actual, &a, None, &executor);
        prop_assert!(ok, "NTT squaring failed");
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_ntt_sqr_with_scratch_matches_executor(a in operand_strategy()) {
        let executor = SequentialExecutor;
        let result_len = a.len().wrapping_mul(2);
        let mut actual_allocating = vec![0_usize; result_len];
        let mut actual_scratch = vec![0_usize; result_len];

        let cap_a = Ntt::digit_capacity(a.len(), 50).expect("capacity fits");
        let transform_len = Ntt::transform_capacity(cap_a, cap_a).expect("transform capacity fits");
        let mut scratch = vec![0.0_f64; Ntt::scratch_sqr_len(transform_len)];

        let ok1 = Ntt::try_sqr_with_executor(&mut actual_allocating, &a, None, &executor);
        let ok2 = Ntt::try_sqr_with_scratch(&mut actual_scratch, &a, &mut scratch, &executor);

        prop_assert!(ok1 && ok2);
        prop_assert_eq!(actual_allocating, actual_scratch);
    }
}
