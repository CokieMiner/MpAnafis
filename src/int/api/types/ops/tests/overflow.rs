//! Operator overflow and destination-preserving assignment failures.

extern crate std;

use core::{
    ops::{
        Add, AddAssign, BitOrAssign, BitXorAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign,
        Sub, SubAssign,
    },
    panic::AssertUnwindSafe,
};
use std::panic::catch_unwind;

use proptest::prelude::{Just, prop_assert, prop_assert_eq, prop_oneof, proptest};

use crate::{BoundedPrecision, MpInt, MpUint};

const INLINE_BITS: usize = 4 * core::mem::size_of::<usize>() * 8;

macro_rules! reject_assignment_paths {
    ($left:expr, $right:expr, $assign:ident, $method:ident) => {{
        let initial = &$left;
        let operand = &$right;
        for owned in [false, true] {
            let mut receiver = initial.clone();
            let failure = catch_unwind(AssertUnwindSafe(|| {
                if owned {
                    $assign::$method(&mut receiver, operand.clone());
                } else {
                    $assign::$method(&mut receiver, operand);
                }
            }));
            prop_assert!(failure.is_err());
            prop_assert_eq!(&receiver, initial);
            prop_assert_eq!(receiver.precision(), initial.precision());
        }
    }};
}

macro_rules! reject_operator_paths {
    ($left:expr, $right:expr, $binary:ident, $assign:ident, $method:ident, $assign_method:ident) => {{
        let left = &$left;
        let right = &$right;
        let operations: [&dyn Fn(); 4] = [
            &|| drop($binary::$method(left, right)),
            &|| drop($binary::$method(left.clone(), right)),
            &|| drop($binary::$method(left, right.clone())),
            &|| drop($binary::$method(left.clone(), right.clone())),
        ];
        for operation in operations {
            prop_assert!(catch_unwind(AssertUnwindSafe(operation)).is_err());
        }
        reject_assignment_paths!(*left, *right, $assign, $assign_method);
    }};
}

proptest! {
    #[test]
    fn bounded_operator_failures_preserve_values_and_destination_precision(
        bits in prop_oneof![
            Just(1_usize), Just(INLINE_BITS.checked_sub(1).expect("nonzero inline width")),
            Just(INLINE_BITS), Just(INLINE_BITS.checked_add(1).expect("small inline width")),
            1_usize..=512,
        ],
    ) {
        let width = BoundedPrecision::new(bits).expect("positive test width");
        let unsigned_maximum = MpUint::max_for_precision(bits);
        let unsigned_zero = MpUint::zero_with_precision(width);
        let unsigned_one = MpUint::with_precision_checked(1_u8, width).expect("one fits");
        reject_operator_paths!(unsigned_maximum, unsigned_one, Add, AddAssign, add, add_assign);
        reject_operator_paths!(unsigned_zero, unsigned_one, Sub, SubAssign, sub, sub_assign);
        reject_operator_paths!(unsigned_maximum, unsigned_zero, Div, DivAssign, div, div_assign);
        reject_operator_paths!(unsigned_maximum, unsigned_zero, Rem, RemAssign, rem, rem_assign);
        if bits > 1 {
            let two = MpUint::with_precision_checked(2_u8, width).expect("two fits at least two bits");
            reject_operator_paths!(unsigned_maximum, two, Mul, MulAssign, mul, mul_assign);
        }
        let signed_maximum = MpInt::max_for_precision(bits);
        let signed_minimum = MpInt::min_for_precision(bits);
        let signed_minus_one = MpInt::with_precision_checked(-1_i8, width).expect("minus one fits");
        let signed_zero = MpInt::zero_with_precision(width);
        reject_operator_paths!(signed_minimum, signed_minus_one, Add, AddAssign, add, add_assign);
        reject_operator_paths!(signed_maximum, signed_minus_one, Sub, SubAssign, sub, sub_assign);
        reject_operator_paths!(signed_minimum, signed_minus_one, Mul, MulAssign, mul, mul_assign);
        reject_operator_paths!(signed_minimum, signed_minus_one, Div, DivAssign, div, div_assign);
        reject_operator_paths!(signed_minimum, signed_minus_one, Rem, RemAssign, rem, rem_assign);
        reject_operator_paths!(signed_minimum, signed_zero, Div, DivAssign, div, div_assign);
        reject_operator_paths!(signed_minimum, signed_zero, Rem, RemAssign, rem, rem_assign);
        if bits > 1 {
            let one = MpInt::with_precision_checked(1_i8, width).expect("one fits at least two signed bits");
            reject_operator_paths!(signed_maximum, one, Add, AddAssign, add, add_assign);
            reject_operator_paths!(signed_minimum, one, Sub, SubAssign, sub, sub_assign);
        }

        let unsigned_high_bit = MpUint::one().checked_shl(bits).expect("unlimited shift");
        let signed_high_bit = MpInt::one().checked_shl(bits.checked_sub(1).expect("positive width")).expect("unlimited shift");
        reject_assignment_paths!(unsigned_maximum, MpUint::one(), AddAssign, add_assign);
        reject_assignment_paths!(unsigned_maximum, MpUint::from(2_u8), MulAssign, mul_assign);
        reject_assignment_paths!(unsigned_zero, unsigned_high_bit, BitOrAssign, bitor_assign);
        reject_assignment_paths!(unsigned_zero, unsigned_high_bit, BitXorAssign, bitxor_assign);
        reject_assignment_paths!(signed_maximum, MpInt::one(), AddAssign, add_assign);
        reject_assignment_paths!(signed_minimum, MpInt::one(), SubAssign, sub_assign);
        reject_assignment_paths!(signed_minimum, MpInt::minus_one(), MulAssign, mul_assign);
        reject_assignment_paths!(signed_zero, signed_high_bit, BitOrAssign, bitor_assign);
        reject_assignment_paths!(signed_zero, signed_high_bit, BitXorAssign, bitxor_assign);
    }
}
