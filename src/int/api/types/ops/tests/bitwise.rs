//! Bitwise precision closure and asymmetric assignment domains.

extern crate std;

use core::{ops::BitAndAssign, panic::AssertUnwindSafe};
use std::panic::catch_unwind;

use proptest::prelude::{any, prop_assert_eq, proptest};

use crate::{BoundedPrecision, MpInt, MpUint, Precision};

proptest! {
    #[test]
    fn signed_binary_bitwise_fits_combined_precision(
        left_seed in any::<i128>(), right_seed in any::<i128>(),
        left_bits in 1_usize..=128, right_bits in 1_usize..=128,
        left_unlimited in any::<bool>(), right_unlimited in any::<bool>(),
    ) {
        let left_shift = u32::try_from(128_usize.checked_sub(left_bits).expect("width at most 128"))
            .expect("shift at most 127");
        let right_shift = u32::try_from(128_usize.checked_sub(right_bits).expect("width at most 128"))
            .expect("shift at most 127");
        // Removing then sign-extending the high bits is an independent i128 oracle.
        let left_native = left_seed.wrapping_shl(left_shift).wrapping_shr(left_shift);
        let right_native = right_seed.wrapping_shl(right_shift).wrapping_shr(right_shift);
        let left_width = BoundedPrecision::new(left_bits).expect("positive test width");
        let right_width = BoundedPrecision::new(right_bits).expect("positive test width");
        let left = if left_unlimited { MpInt::from(left_native) } else { MpInt::with_precision_checked(left_native, left_width).expect("signed residue fits") };
        let right = if right_unlimited { MpInt::from(right_native) } else { MpInt::with_precision_checked(right_native, right_width).expect("signed residue fits") };
        let precision = if left_unlimited || right_unlimited {
            Precision::Unlimited
        } else {
            Precision::new_bounded(left_bits.max(right_bits)).expect("valid test width")
        };
        for (result, expected) in [
            (&left & &right, left_native & right_native),
            (left.clone() & &right, left_native & right_native),
            (&left & right.clone(), left_native & right_native),
            (left.clone() & right.clone(), left_native & right_native),
            (&left | &right, left_native | right_native),
            (left.clone() | &right, left_native | right_native),
            (&left | right.clone(), left_native | right_native),
            (left.clone() | right.clone(), left_native | right_native),
            (&left ^ &right, left_native ^ right_native),
            (left.clone() ^ &right, left_native ^ right_native),
            (&left ^ right.clone(), left_native ^ right_native),
            (left.clone() ^ right, left_native ^ right_native),
        ] {
            prop_assert_eq!(result.to_i128(), Some(expected));
            prop_assert_eq!(result.precision(), precision);
        }
        for result in [!&left, !left.clone()] {
            prop_assert_eq!(result.to_i128(), Some(!left_native));
            prop_assert_eq!(result.precision(), left.precision());
        }
    }

    #[test]
    fn unsigned_binary_bitwise_fits_combined_precision(
        left_seed in any::<u128>(), right_seed in any::<u128>(),
        left_bits in 1_usize..=128, right_bits in 1_usize..=128,
        left_unlimited in any::<bool>(), right_unlimited in any::<bool>(),
    ) {
        let left_mask = u128::MAX >> 128_usize.checked_sub(left_bits).expect("width at most 128");
        let right_mask = u128::MAX >> 128_usize.checked_sub(right_bits).expect("width at most 128");
        let left_native = left_seed & left_mask;
        let right_native = right_seed & right_mask;
        let left_width = BoundedPrecision::new(left_bits).expect("positive test width");
        let right_width = BoundedPrecision::new(right_bits).expect("positive test width");
        let left = if left_unlimited { MpUint::from(left_native) } else { MpUint::with_precision_checked(left_native, left_width).expect("unsigned residue fits") };
        let right = if right_unlimited { MpUint::from(right_native) } else { MpUint::with_precision_checked(right_native, right_width).expect("unsigned residue fits") };
        let precision = if left_unlimited || right_unlimited {
            Precision::Unlimited
        } else {
            Precision::new_bounded(left_bits.max(right_bits)).expect("valid test width")
        };
        for (result, expected) in [
            (&left & &right, left_native & right_native),
            (left.clone() & &right, left_native & right_native),
            (&left & right.clone(), left_native & right_native),
            (left.clone() & right.clone(), left_native & right_native),
            (&left | &right, left_native | right_native),
            (left.clone() | &right, left_native | right_native),
            (&left | right.clone(), left_native | right_native),
            (left.clone() | right.clone(), left_native | right_native),
            (&left ^ &right, left_native ^ right_native),
            (left.clone() ^ &right, left_native ^ right_native),
            (&left ^ right.clone(), left_native ^ right_native),
            (left.clone() ^ right.clone(), left_native ^ right_native),
        ] {
            prop_assert_eq!(result.to_u128(), Some(expected));
            prop_assert_eq!(result.precision(), precision);
        }
        let mut destination = left.clone();
        destination &= &right;
        prop_assert_eq!(destination.to_u128(), Some(left_native & right_native));
        prop_assert_eq!(destination.precision(), left.precision());
        if !left_unlimited {
            for result in [!&left, !left.clone(), left.try_not().expect("bounded width")] {
                prop_assert_eq!(result.to_u128(), Some(!left_native & left_mask));
                prop_assert_eq!(result.precision(), left.precision());
            }
        }
    }
}

#[test]
fn bitwise_precision_closure_at_inline_and_heap_boundaries() {
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    let inline_bits = limb_bits.checked_mul(4).expect("small inline width");
    for bits in [
        1,
        limb_bits,
        inline_bits
            .checked_sub(1)
            .expect("four limbs exceed one bit"),
        inline_bits,
        inline_bits.checked_add(1).expect("small heap width"),
    ] {
        let width = BoundedPrecision::new(bits).expect("valid test width");
        let minimum = MpInt::min_for_precision(bits);
        let maximum = MpInt::max_for_precision(bits);
        let minus_one = MpInt::with_precision_checked(-1_i8, width).expect("-1 fits every width");
        let zero = MpInt::zero_with_precision(width);
        for (result, expected) in [
            (&minimum & &maximum, &zero),
            (&minimum | &maximum, &minus_one),
            (&minimum ^ &maximum, &minus_one),
            (!&minimum, &maximum),
            (!&maximum, &minimum),
        ] {
            assert_eq!(&result, expected);
            assert_eq!(result.precision(), Precision::Bounded(width));
        }
        let unsigned_maximum = MpUint::max_for_precision(bits);
        let unsigned_zero = !&unsigned_maximum;
        assert!(unsigned_zero.is_zero());
        assert_eq!(unsigned_zero.precision(), Precision::Bounded(width));
        assert_eq!(!unsigned_zero, unsigned_maximum);
    }
}

#[test]
fn signed_and_assignment_keeps_narrow_destination_transactional() {
    let initial =
        MpInt::with_precision_checked(-1_i8, BoundedPrecision::new(1).expect("valid test width"))
            .expect("minus one fits");
    let rhs =
        MpInt::with_precision_checked(1_i8, BoundedPrecision::new(2).expect("valid test width"))
            .expect("one fits two signed bits");
    // -1 & 1 is +1, which fits the combined width but not a one-bit receiver.
    assert_eq!((&initial & &rhs).to_i128(), Some(1));
    for owned in [false, true] {
        let mut destination = initial.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if owned {
                BitAndAssign::bitand_assign(&mut destination, rhs.clone());
            } else {
                BitAndAssign::bitand_assign(&mut destination, &rhs);
            }
        }));
        assert!(outcome.is_err(), "the one-bit receiver cannot represent +1");
        assert_eq!(destination, initial);
        assert_eq!(destination.precision(), initial.precision());
    }
}
