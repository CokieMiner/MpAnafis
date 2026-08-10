//! Assignment-operator precision properties.

use core::ops::{
    AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, DivAssign, MulAssign, RemAssign, SubAssign,
};

use proptest::prelude::*;

use super::{ArbiInt, ArbiUint, bounded_int, bounded_uint};
use crate::Precision;

proptest! {
    #[test]
    fn unsigned_assignment_preserves_lhs_precision(
        bits in 8_usize..=64,
        lhs_value in 0_u8..=127,
        rhs_value in 1_u8..=127,
        small_lhs in 0_u8..=15,
        small_rhs in 1_u8..=15,
    ) {
        let precision = Precision::new_bounded(bits).expect("property width is valid");
        let lhs = bounded_uint(u128::from(lhs_value), bits);
        let rhs = ArbiUint::from(rhs_value);

        let mut sum_ref = lhs.clone();
        AddAssign::add_assign(&mut sum_ref, &rhs);
        prop_assert_eq!(sum_ref.precision, precision);

        let mut sum_owned = lhs.clone();
        AddAssign::add_assign(&mut sum_owned, rhs.clone());
        prop_assert_eq!(sum_owned.precision, precision);

        let (larger, smaller) = if lhs_value >= rhs_value {
            (lhs_value, rhs_value)
        } else {
            (rhs_value, lhs_value)
        };
        let mut difference = bounded_uint(u128::from(larger), bits);
        SubAssign::sub_assign(&mut difference, ArbiUint::from(smaller));
        prop_assert_eq!(difference.precision, precision);

        let mut product = bounded_uint(u128::from(small_lhs), bits);
        MulAssign::mul_assign(&mut product, ArbiUint::from(small_rhs));
        prop_assert_eq!(product.precision, precision);

        let mut quotient = lhs.clone();
        DivAssign::div_assign(&mut quotient, &rhs);
        prop_assert_eq!(quotient.precision, precision);

        let mut remainder = lhs.clone();
        RemAssign::rem_assign(&mut remainder, rhs.clone());
        prop_assert_eq!(remainder.precision, precision);

        let mut and_value = lhs.clone();
        BitAndAssign::bitand_assign(&mut and_value, &rhs);
        prop_assert_eq!(and_value.precision, precision);

        let mut or_value = lhs.clone();
        BitOrAssign::bitor_assign(&mut or_value, rhs.clone());
        prop_assert_eq!(or_value.precision, precision);

        let mut xor_value = lhs;
        BitXorAssign::bitxor_assign(&mut xor_value, rhs);
        prop_assert_eq!(xor_value.precision, precision);
    }

    #[test]
    fn signed_assignment_preserves_lhs_precision(
        bits in 8_usize..=64,
        lhs_value in -31_i8..=31,
        rhs_value in 1_i8..=31,
        small_lhs in -10_i8..=10,
        small_rhs in 1_i8..=10,
    ) {
        let precision = Precision::new_bounded(bits).expect("property width is valid");
        let lhs = bounded_int(i128::from(lhs_value), bits);
        let rhs = ArbiInt::from(rhs_value);

        let mut sum_ref = lhs.clone();
        AddAssign::add_assign(&mut sum_ref, &rhs);
        prop_assert_eq!(sum_ref.precision, precision);

        let mut sum_owned = lhs.clone();
        AddAssign::add_assign(&mut sum_owned, rhs.clone());
        prop_assert_eq!(sum_owned.precision, precision);

        let mut difference = lhs.clone();
        SubAssign::sub_assign(&mut difference, rhs.clone());
        prop_assert_eq!(difference.precision, precision);

        let mut product = bounded_int(i128::from(small_lhs), bits);
        MulAssign::mul_assign(&mut product, ArbiInt::from(small_rhs));
        prop_assert_eq!(product.precision, precision);

        let mut quotient = lhs.clone();
        DivAssign::div_assign(&mut quotient, &rhs);
        prop_assert_eq!(quotient.precision, precision);

        let mut remainder = lhs.clone();
        RemAssign::rem_assign(&mut remainder, rhs.clone());
        prop_assert_eq!(remainder.precision, precision);

        let mut and_value = lhs.clone();
        BitAndAssign::bitand_assign(&mut and_value, &rhs);
        prop_assert_eq!(and_value.precision, precision);

        let mut or_value = lhs.clone();
        BitOrAssign::bitor_assign(&mut or_value, rhs.clone());
        prop_assert_eq!(or_value.precision, precision);

        let mut xor_value = lhs;
        BitXorAssign::bitxor_assign(&mut xor_value, rhs);
        prop_assert_eq!(xor_value.precision, precision);
    }
}
