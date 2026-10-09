//! Fused assignment destination precision and transactional failure contracts.

extern crate std;

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::{vec, vec::Vec};

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{BoundedPrecision, MpInt, MpUint, Precision};

proptest! {
    #[test]
    fn unsigned_fused_results_obey_destination_precision(
        bits in 1_usize..=16,
        a in any::<u8>(),
        b in any::<u8>(),
        seed in any::<u8>(),
        bounded_operands in any::<bool>(),
        unlimited_destination in any::<bool>(),
    ) {
        let operand_width = BoundedPrecision::new(8).expect("valid byte width");
        let destination_width = BoundedPrecision::new(bits).expect("positive test width");
        let left = if bounded_operands { MpUint::with_precision_checked(a, operand_width).expect("byte fits") } else { MpUint::from(a) };
        let right = if bounded_operands { MpUint::with_precision_checked(b, operand_width).expect("byte fits") } else { MpUint::from(b) };
        let max = (1_u128 << bits).checked_sub(1).expect("positive power of two");
        let original = if unlimited_destination {
            MpUint::from(seed)
        } else {
            MpUint::with_precision_checked(u128::from(seed) & max, destination_width).expect("residue fits")
        };
        let wide_a = u128::from(a);
        let wide_b = u128::from(b);
        let results = [
            Some(wide_a.checked_add(wide_b).expect("u8 sum fits u128")),
            wide_a.checked_sub(wide_b),
            Some(wide_a.checked_mul(wide_b).expect("u8 product fits u128")),
            Some(wide_a.checked_mul(wide_a).expect("u8 square fits u128")),
        ];
        for (operation, exact) in results.into_iter().enumerate() {
            let mut destination = original.clone();
            destination.reserve(32);
            let capacity = destination.capacity();
            let outcome = catch_unwind(AssertUnwindSafe(|| match operation {
                0 => { destination.assign_add(&left, &right); false }
                1 => destination.assign_sub(&left, &right),
                2 => { destination.assign_mul(&left, &right); false }
                _ => { destination.assign_square(&left); false }
            }));
            match exact {
                None => {
                    prop_assert_eq!(outcome.expect("underflow reports a flag"), true);
                    prop_assert_eq!(&destination, &original);
                }
                Some(value) if unlimited_destination || value <= max => {
                    prop_assert_eq!(outcome.expect("representable assignment succeeds"), false);
                    prop_assert_eq!(&destination, &MpUint::from(value));
                }
                Some(_) => {
                    prop_assert!(outcome.is_err());
                    prop_assert_eq!(&destination, &original);
                }
            }
            prop_assert_eq!(destination.precision(), original.precision());
            prop_assert_eq!(destination.capacity(), capacity);
        }
    }

    #[test]
    fn signed_fused_results_obey_destination_precision(
        bits in 1_usize..=17,
        a in any::<i8>(),
        b in any::<i8>(),
        bounded_operands in any::<bool>(),
        unlimited_destination in any::<bool>(),
    ) {
        let operand_width = BoundedPrecision::new(8).expect("valid byte width");
        let destination_width = BoundedPrecision::new(bits).expect("positive test width");
        let left = if bounded_operands { MpInt::with_precision_checked(a, operand_width).expect("signed byte fits") } else { MpInt::from(a) };
        let right = if bounded_operands { MpInt::with_precision_checked(b, operand_width).expect("signed byte fits") } else { MpInt::from(b) };
        let limit = 1_i128 << bits.checked_sub(1).expect("nonzero precision");
        let original = if unlimited_destination { MpInt::minus_one() } else { MpInt::with_precision_checked(-1_i8, destination_width).expect("minus one fits") };
        let wide_a = i128::from(a);
        let wide_b = i128::from(b);
        let results = [
            wide_a.checked_add(wide_b).expect("i8 sum fits i128"),
            wide_a.checked_sub(wide_b).expect("i8 difference fits i128"),
            wide_a.checked_mul(wide_b).expect("i8 product fits i128"),
            wide_a.checked_mul(wide_a).expect("i8 square fits i128"),
        ];
        for (operation, exact) in results.into_iter().enumerate() {
            let mut destination = original.clone();
            destination.reserve(32);
            let capacity = destination.capacity();
            let outcome = catch_unwind(AssertUnwindSafe(|| match operation {
                0 => destination.assign_add(&left, &right),
                1 => destination.assign_sub(&left, &right),
                2 => destination.assign_mul(&left, &right),
                _ => destination.assign_square(&left),
            }));
            if unlimited_destination || (limit.checked_neg().expect("small positive bound")..limit).contains(&exact) {
                prop_assert!(outcome.is_ok());
                prop_assert_eq!(&destination, &MpInt::from(exact));
            } else {
                prop_assert!(outcome.is_err());
                prop_assert_eq!(&destination, &original);
            }
            prop_assert_eq!(destination.precision(), original.precision());
            prop_assert_eq!(destination.capacity(), capacity);
        }
    }
}

#[test]
fn fused_assignment_accepts_large_operands_when_the_result_fits() {
    let width = BoundedPrecision::new(1).expect("valid test width");
    let large = MpUint::one() << 1024_usize;
    let mut unsigned = MpUint::with_precision_checked(1_u8, width).expect("one fits");
    assert!(!unsigned.assign_sub(&large, &large));
    assert!(unsigned.is_zero());
    unsigned.assign_mul(&large, &MpUint::zero());
    assert!(unsigned.is_zero());
    assert_eq!(unsigned.precision(), Precision::Bounded(width));

    let positive = MpInt::from(large);
    let negative = -&positive;
    let mut signed = MpInt::with_precision_checked(-1_i8, width).expect("minus one fits");
    signed.assign_add(&positive, &negative);
    assert!(signed.is_zero());
    assert!(!signed.is_negative());
    signed.assign_sub(&negative, &negative);
    assert!(signed.is_zero());
    signed.assign_mul(&negative, &MpInt::zero());
    assert!(signed.is_zero());
    assert_eq!(signed.precision(), Precision::Bounded(width));
}

#[test]
fn bounded_sum_cancels_without_requiring_representable_operand_bit_counts() {
    let width = BoundedPrecision::new(1).expect("valid test width");
    // On 16-bit targets this addressable magnitude has 65,537 significant
    // bits, exceeding usize::MAX. Its sum with its negation still fits one bit.
    let mut limbs = vec![0_usize; 4097];
    *limbs.last_mut().expect("nonempty magnitude") = 1;
    let bytes: Vec<_> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    let positive = MpInt::from(MpUint::zero() + MpUint::from_le_bytes(&bytes));
    let negative = -&positive;
    let mut destination = MpInt::with_precision_checked(-1_i8, width).expect("minus one fits");
    destination.assign_add(&positive, &negative);
    assert!(destination.is_zero());
    assert!(!destination.is_negative());
    assert_eq!(destination.precision(), Precision::Bounded(width));
}

#[test]
fn bounded_heap_products_preserve_buffers_and_roll_back_overflow() {
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    let bits = 8 * limb_bits;
    let precision = Precision::new_bounded(bits).expect("eight limbs is a valid width");
    let overflow_operand = MpUint::one() << bits.div_euclid(2);
    let operand = &overflow_operand - MpUint::one();
    let expected = &operand * &operand;
    let width = BoundedPrecision::new(bits).expect("valid bounded width");
    let mut unsigned = MpUint::zero_with_precision(width);
    unsigned.reserve(32);
    let capacity = unsigned.capacity();
    unsigned.assign_mul(&operand, &operand);
    assert_eq!(unsigned, expected);
    unsigned.assign_square(&operand);
    assert_eq!(unsigned, expected);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            unsigned.assign_square(&overflow_operand);
        }))
        .is_err()
    );
    assert_eq!(unsigned, expected);
    assert_eq!(unsigned.precision(), precision);
    assert_eq!(unsigned.capacity(), capacity);

    let negative_operand = -MpInt::from(operand >> 1_usize);
    let square_expected = &negative_operand * &negative_operand;
    let mut signed = MpInt::zero_with_precision(width);
    signed.reserve(32);
    let signed_capacity = signed.capacity();
    signed.assign_square(&negative_operand);
    assert_eq!(signed, square_expected);
    signed.assign_mul(&negative_operand, &(-&negative_operand));
    let negative_expected = -square_expected;
    assert_eq!(signed, negative_expected);
    let signed_overflow_operand = MpInt::from(overflow_operand);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            signed.assign_mul(&signed_overflow_operand, &signed_overflow_operand);
        }))
        .is_err()
    );
    assert_eq!(signed, negative_expected);
    assert_eq!(signed.precision(), precision);
    assert_eq!(signed.capacity(), signed_capacity);
}

#[test]
fn fused_assignments_match_assignment_operators_at_a_limb_boundary() {
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    for width in [limb_bits - 1, limb_bits, limb_bits + 1] {
        let precision = BoundedPrecision::new(width).expect("limb boundary width is valid");
        let left = MpUint::max_for_precision(width);
        let right = MpUint::one();
        let mut expected = left.clone();
        expected -= &right;
        let mut actual = MpUint::zero_with_precision(precision);
        assert!(!actual.assign_sub(&left, &right));
        assert_eq!(actual, expected);
        assert_eq!(actual.precision(), expected.precision());

        let signed_left = MpInt::min_for_precision(width);
        let signed_right = MpInt::one();
        let mut signed_expected = signed_left.clone();
        signed_expected += &signed_right;
        let mut signed_actual = MpInt::zero_with_precision(precision);
        signed_actual.assign_add(&signed_left, &signed_right);
        assert_eq!(signed_actual, signed_expected);
        assert_eq!(signed_actual.precision(), signed_expected.precision());
    }
}
