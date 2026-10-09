//! Infinite two's-complement identities across sign quadrants and storage classes.

use core::ops::{BitAnd, BitOr, BitXor};

use alloc::vec;

use proptest::{
    prelude::any,
    test_runner::{Config, TestRunner},
};

use crate::int::{INLINE_LIMBS, InternalMpInt, InternalMpUint, LIMB_BITS, Limb};

use super::strategies::{public, signed};

#[test]
fn bitwise_operations_match_native_and_finite_twos_complement() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(&(any::<i128>(), any::<i128>()), |(a, b)| {
            let left = InternalMpInt {
                abs: InternalMpUint::from_u128(a.unsigned_abs()),
                is_positive: a >= 0,
            };
            let right = InternalMpInt {
                abs: InternalMpUint::from_u128(b.unsigned_abs()),
                is_positive: b >= 0,
            };
            for (actual, expected) in [
                (&left & &right, a & b),
                (&left | &right, a | b),
                (&left ^ &right, a ^ b),
                (!&left, !a),
            ] {
                assert_eq!(public(&actual).to_i128(), Some(expected));
            }
            Ok(())
        })
        .expect("primitive bitwise results agree");
    let widths: &[usize] = if cfg!(miri) {
        &[1, 4, 5]
    } else {
        &[1, 3, 4, 5, 8, 64]
    };
    for &left_len in widths {
        for &right_len in widths {
            for left_positive in [false, true] {
                for right_positive in [false, true] {
                    let left = InternalMpInt {
                        abs: InternalMpUint::from_limbs(vec![Limb::MAX.div_euclid(3); left_len]),
                        is_positive: left_positive,
                    };
                    let right = InternalMpInt {
                        abs: InternalMpUint::from_limbs(vec![
                            Limb::MAX.div_euclid(3).rotate_left(1);
                            right_len
                        ]),
                        is_positive: right_positive,
                    };
                    check_bitwise(&left, &right);
                }
            }
        }
    }
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
            ),
            |(left, right)| {
                check_bitwise(&left, &right);
                Ok(())
            },
        )
        .expect("wide identities and ownership forms agree");
}

#[test]
fn negative_carries_and_cancellation_produce_compact_zero_and_minus_one() {
    let minus_one = InternalMpInt {
        abs: InternalMpUint::one(),
        is_positive: false,
    };
    for limbs in [1_usize, 4, 5, 8, 64] {
        let width = limbs.checked_mul(LIMB_BITS).expect("small test width");
        let power = InternalMpInt {
            abs: InternalMpUint::power_of_two(width),
            is_positive: false,
        };
        let canceled = &power ^ &power;
        assert_eq!(canceled, InternalMpInt::zero());
        assert_eq!(canceled.abs.capacity(), INLINE_LIMBS);
        let complement = !&power;
        for union in [
            &power | &complement,
            &complement | &power,
            &power | &minus_one,
        ] {
            assert_eq!(union, minus_one);
            assert_eq!(union.abs.capacity(), INLINE_LIMBS);
        }
        let mut all_ones = power.abs.clone();
        all_ones.decrement();
        let negative_all_ones = InternalMpInt {
            abs: all_ones,
            is_positive: false,
        };
        check_bitwise(&power, &negative_all_ones);
        check_bitwise(&negative_all_ones, &minus_one);
    }
}

fn check_bitwise(left: &InternalMpInt, right: &InternalMpInt) {
    let width = left
        .required_signed_bits_for_bounded_storage()
        .max(right.required_signed_bits_for_bounded_storage())
        .checked_add(1)
        .expect("test magnitudes leave an extra sign bit");
    let a = left.to_tc_bits(width);
    let b = right.to_tc_bits(width);
    let and = left & right;
    let or = left | right;
    let xor = left ^ right;
    for (actual, encoded) in [
        (&and, a.bitand(&b)),
        (&or, a.bitor(&b)),
        (&xor, a.bitxor(&b)),
    ] {
        assert_eq!(*actual, InternalMpInt::from_tc_bits(encoded, width));
        assert!(actual.is_positive || !actual.abs.is_zero());
    }
    assert_eq!(left.clone() & right.clone(), and);
    assert_eq!(left.clone() & right, and);
    assert_eq!(left & right.clone(), and);
    assert_eq!(left.clone() | right.clone(), or);
    assert_eq!(left.clone() | right, or);
    assert_eq!(left | right.clone(), or);
    assert_eq!(left.clone() ^ right.clone(), xor);
    assert_eq!(left.clone() ^ right, xor);
    assert_eq!(left ^ right.clone(), xor);
    assert_eq!(right & left, and);
    assert_eq!(right | left, or);
    assert_eq!(right ^ left, xor);
    check_bitwise_identities(left, right);
}

fn check_bitwise_identities(left: &InternalMpInt, right: &InternalMpInt) {
    assert_eq!(BitAnd::bitand(left, left), *left);
    assert_eq!(BitOr::bitor(left, left), *left);
    assert_eq!(BitXor::bitxor(left, left), InternalMpInt::zero());
    assert_eq!(left & &InternalMpInt::zero(), InternalMpInt::zero());
    assert_eq!(left | &InternalMpInt::zero(), *left);
    assert_eq!(left ^ &InternalMpInt::zero(), *left);
    let minus_one = InternalMpInt {
        abs: InternalMpUint::one(),
        is_positive: false,
    };
    assert_eq!(left & &minus_one, *left);
    assert_eq!(left | &minus_one, minus_one);
    assert_eq!(left ^ &minus_one, !left);
    assert_eq!(!left.clone(), !left);
    assert_eq!(!(!left), *left);
    assert_eq!(!(!left.clone()), *left);
    assert_eq!(!(left & right), (!left) | (!right));
    assert_eq!(!(left | right), (!left) & (!right));
    assert_eq!(left ^ right, (left & (!right)) | ((!left) & right));
}
