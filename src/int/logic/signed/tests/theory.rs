//! Signed Bézout identities and Jacobi symbols against binary reciprocity.

use core::mem;

use proptest::test_runner::{Config, TestRunner};

use crate::int::{InternalMpInt, InternalMpUint};

use super::strategies::{public, signed, small_signed};

#[test]
fn bezout_coefficients_include_input_signs_and_jacobi_matches_reciprocity() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(
                signed(if cfg!(miri) { 6 } else { 16 }),
                signed(if cfg!(miri) { 6 } else { 16 }),
            ),
            |(left, generated_right)| {
                let right = if generated_right.abs.is_zero() {
                    InternalMpInt::one()
                } else {
                    generated_right
                };
                let (gcd, x, y) = left.extended_gcd(&right);
                assert_eq!(left.mul(&x).add(&right.mul(&y)), gcd);
                assert_eq!(public(&gcd), public(&left).gcd(&public(&right)));
                assert!(gcd.is_positive);
                assert!(!gcd.abs.is_zero());
                assert!(x.is_positive || !x.abs.is_zero());
                assert!(y.is_positive || !y.abs.is_zero());
                Ok(())
            },
        )
        .expect("wide signed Bezout coefficients reconstruct a positive gcd");
    for numerator in [i64::MIN, -17, -1, 0, 1, 17, i64::MAX] {
        for modulus in [1_u64, 3, 5, 7, 9, 15, 101] {
            let value = InternalMpInt {
                abs: InternalMpUint::from_u128(u128::from(numerator.unsigned_abs())),
                is_positive: numerator >= 0,
            };
            let divisor = InternalMpInt {
                abs: InternalMpUint::from_u128(u128::from(modulus)),
                is_positive: true,
            };
            assert_eq!(
                value.jacobi_symbol(&divisor),
                binary_jacobi(numerator, modulus)
            );
        }
    }
    cases
        .run(
            &(small_signed(), 0_u64..=5000),
            |((native, value), half_modulus)| {
                let modulus = half_modulus
                    .checked_mul(2)
                    .and_then(|even| even.checked_add(1))
                    .expect("small positive odd modulus");
                let divisor = InternalMpInt {
                    abs: InternalMpUint::from_u128(u128::from(modulus)),
                    is_positive: true,
                };
                assert_eq!(
                    value.jacobi_symbol(&divisor),
                    binary_jacobi(native, modulus)
                );
                Ok(())
            },
        )
        .expect("signed Jacobi agrees with an independent reciprocity reduction");
}

fn binary_jacobi(numerator: i64, modulus: u64) -> i8 {
    let signed_modulus = i128::from(modulus);
    let mut a = u64::try_from(i128::from(numerator).rem_euclid(signed_modulus))
        .expect("positive residue fits u64");
    let mut m = modulus;
    let mut symbol = 1_i8;
    while a != 0 {
        let twos = a.trailing_zeros();
        a = a
            .checked_shr(twos)
            .expect("nonzero input has fewer than 64 trailing zeros");
        if twos.rem_euclid(2) == 1 && matches!(m.rem_euclid(8), 3 | 5) {
            symbol = symbol.wrapping_neg();
        }
        mem::swap(&mut a, &mut m);
        if a.rem_euclid(4) == 3 && m.rem_euclid(4) == 3 {
            symbol = symbol.wrapping_neg();
        }
        a = a.rem_euclid(m);
    }
    if m == 1 { symbol } else { 0 }
}
