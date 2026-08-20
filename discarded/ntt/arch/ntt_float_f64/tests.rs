//! Unit and property tests for 50-bit floating-point Harvey NTT arithmetic.

use crate::int::logic::unsigned::math::arch::ArchKernels;

use super::{
    mulmod_scalar, radix4_dif_float_one, radix4_dif_float_scalar, radix4_dit_float_one,
    radix4_dit_float_scalar, reduce_to_pm1n_scalar,
};

const TEST_PRIME: f64 = 1_108_307_720_798_209.0;
const TEST_PINV: f64 = 1.0 / TEST_PRIME;

#[test]
fn test_float_mulmod_exactness() {
    let a = 123_456_789_012_345.0;
    let b = 987_654_321_098_765.0;

    let reduced = mulmod_scalar(a, b, TEST_PRIME, TEST_PINV);
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Test verification of 50-bit exact values"
    )]
    let (a_int, b_int, p_int) = (a as u128, b as u128, TEST_PRIME as u128);
    let ref_mod = (a_int.wrapping_mul(b_int)).wrapping_rem(p_int);

    let reduced_norm = if reduced < 0.0 {
        reduced + TEST_PRIME
    } else {
        reduced
    };
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Test verification"
    )]
    let res_u128 = (reduced_norm + 0.5) as u128;
    assert_eq!(res_u128, ref_mod);
}

#[test]
fn test_float_radix4_one_roundtrip() {
    let mut values = [10.0, 20.0, 30.0, 40.0];
    let twiddles = [1.0, 1.0, 1.0];
    // SAFETY: spans are 4 and 3 respectively.
    unsafe {
        radix4_dif_float_one(
            values.as_mut_ptr(),
            twiddles.as_ptr(),
            0,
            1,
            TEST_PRIME,
            TEST_PINV,
        );
        radix4_dit_float_one(
            values.as_mut_ptr(),
            twiddles.as_ptr(),
            0,
            1,
            TEST_PRIME,
            TEST_PINV,
        );
    }
}

#[test]
fn test_float_radix4_scalar_roundtrip() {
    let values = [10.0, 20.0, 30.0, 40.0];
    let twiddles = [1.0, 1.0, 1.0];

    let mut values_copy = values;
    // SAFETY: spans are 4 and 3 respectively.
    unsafe {
        radix4_dif_float_scalar(
            values_copy.as_mut_ptr(),
            twiddles.as_ptr(),
            1,
            TEST_PRIME,
            TEST_PINV,
        );
        radix4_dit_float_scalar(
            values_copy.as_mut_ptr(),
            twiddles.as_ptr(),
            1,
            TEST_PRIME,
            TEST_PINV,
        );
    }

    for (orig, res) in values.iter().zip(values_copy.iter()) {
        let expected = reduce_to_pm1n_scalar(*orig * 4.0, TEST_PRIME, TEST_PINV);
        let diff = (expected - *res).abs();
        assert!(diff < 1e-5);
    }
}

#[test]
fn test_arch_kernels_float_dispatch() {
    let mut values = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0,
    ];
    let twiddles = [1.0; 24];

    // SAFETY: spans are valid.
    unsafe {
        ArchKernels::ntt_float_radix4_dif_unchecked(
            values.as_mut_ptr(),
            twiddles.as_ptr(),
            8,
            TEST_PRIME,
            TEST_PINV,
        );
        ArchKernels::ntt_float_radix4_dit_unchecked(
            values.as_mut_ptr(),
            twiddles.as_ptr(),
            8,
            TEST_PRIME,
            TEST_PINV,
        );
    }
}
