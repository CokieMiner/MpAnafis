//! Signed arithmetic against independent primitive results and wide public values.

use proptest::test_runner::{Config, TestRunner};

use crate::int::InternalMpInt;

use super::strategies::{public, signed, small_signed};

#[test]
fn arithmetic_preserves_values_and_canonical_signs_across_storage_widths() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(small_signed(), small_signed()),
            |((a, left), (b, right))| {
                let a_wide = i128::from(a);
                let b_wide = i128::from(b);
                for (result, expected) in [
                    (
                        left.add(&right),
                        a_wide.checked_add(b_wide).expect("i64 sum fits i128"),
                    ),
                    (
                        left.sub(&right),
                        a_wide
                            .checked_sub(b_wide)
                            .expect("i64 difference fits i128"),
                    ),
                    (
                        left.mul(&right),
                        a_wide.checked_mul(b_wide).expect("i64 product fits i128"),
                    ),
                    (
                        left.square(),
                        a_wide.checked_mul(a_wide).expect("i64 square fits i128"),
                    ),
                    (
                        left.mul_into(right),
                        a_wide.checked_mul(b_wide).expect("i64 product fits i128"),
                    ),
                ] {
                    assert_eq!(public(&result).to_i128(), Some(expected));
                    assert!(result.is_positive || !result.abs.is_zero());
                }
                Ok(())
            },
        )
        .expect("primitive arithmetic agrees");
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
                1_usize..=512,
            ),
            |(left, right, bits)| {
                let a = public(&left);
                let b = public(&right);
                for (result, expected) in [
                    (left.add(&right), a.checked_add(&b).expect("unlimited sum")),
                    (
                        left.sub(&right),
                        a.checked_sub(&b).expect("unlimited difference"),
                    ),
                    (
                        left.mul(&right),
                        a.checked_mul(&b).expect("unlimited product"),
                    ),
                    (left.square(), a.checked_mul(&a).expect("unlimited square")),
                    (
                        left.clone().mul_into(right.clone()),
                        a.checked_mul(&b).expect("unlimited owned product"),
                    ),
                ] {
                    assert_eq!(public(&result), expected);
                    assert!(result.is_positive || !result.abs.is_zero());
                }
                assert_eq!(left.add(&InternalMpInt::zero()), left);
                assert_eq!(left.sub(&left), InternalMpInt::zero());
                if left.sum_fits_by_width(&right, bits) {
                    assert!(left.add(&right).required_signed_bits_for_bounded_storage() <= bits);
                }
                Ok(())
            },
        )
        .expect("wide arithmetic agrees and width proofs are sufficient");
}
