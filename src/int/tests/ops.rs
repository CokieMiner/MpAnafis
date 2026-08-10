//! Ownership combinations and assignment behavior for operator traits.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "operator overloads are the API under test; clones test by-value paths"
)]

use core::{ops::*, panic::AssertUnwindSafe};

use proptest::prelude::*;

use super::{std::panic::catch_unwind, strategies};
use crate::int::api::{ArbiInt, ArbiUint, BoundedPrecision};

proptest! {
    #[test]
    fn prop_uint_ops_mul(a in strategies::uint(16), b in strategies::uint(16)) {
        let expected = Mul::mul(a.clone(), &b);
        assert_eq!(Mul::mul(&a, &b), expected);
        assert_eq!(Mul::mul(a.clone(), b.clone()), expected);
        assert_eq!(Mul::mul(&a, b.clone()), expected);

        let mut a1 = a.clone();
        MulAssign::mul_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        MulAssign::mul_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_bitand(a in strategies::uint(32), b in strategies::uint(32)) {
        let expected = BitAnd::bitand(a.clone(), &b);
        assert_eq!(BitAnd::bitand(&a, &b), expected);
        assert_eq!(BitAnd::bitand(a.clone(), b.clone()), expected);
        assert_eq!(BitAnd::bitand(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitAndAssign::bitand_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitAndAssign::bitand_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_bitor(a in strategies::uint(32), b in strategies::uint(32)) {
        let expected = BitOr::bitor(a.clone(), &b);
        assert_eq!(BitOr::bitor(&a, &b), expected);
        assert_eq!(BitOr::bitor(a.clone(), b.clone()), expected);
        assert_eq!(BitOr::bitor(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitOrAssign::bitor_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitOrAssign::bitor_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_bitxor(a in strategies::uint(32), b in strategies::uint(32)) {
        let expected = BitXor::bitxor(a.clone(), &b);
        assert_eq!(BitXor::bitxor(&a, &b), expected);
        assert_eq!(BitXor::bitxor(a.clone(), b.clone()), expected);
        assert_eq!(BitXor::bitxor(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitXorAssign::bitxor_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitXorAssign::bitxor_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_sub(a in strategies::uint(32), b in strategies::uint(32)) {
        if let Some(expected) = a.checked_sub(&b) {
            assert_eq!(Sub::sub(&a, &b), expected);
            assert_eq!(Sub::sub(a.clone(), &b), expected);
            assert_eq!(Sub::sub(&a, b.clone()), expected);
            assert_eq!(Sub::sub(a.clone(), b.clone()), expected);

            let mut a1 = a.clone();
            SubAssign::sub_assign(&mut a1, &b);
            assert_eq!(a1, expected);

            let mut a2 = a;
            SubAssign::sub_assign(&mut a2, b.clone());
            assert_eq!(a2, expected);
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_div(
        a in strategies::uint(32),
        b in strategies::uint_nonzero(32),
    ) {
        if let Some(expected) = a.checked_div(&b) {
            assert_eq!(Div::div(&a, &b), expected);
            assert_eq!(Div::div(a.clone(), &b), expected);
            assert_eq!(Div::div(&a, b.clone()), expected);
            assert_eq!(Div::div(a.clone(), b.clone()), expected);

            let mut a1 = a.clone();
            DivAssign::div_assign(&mut a1, &b);
            assert_eq!(a1, expected);

            let mut a2 = a;
            DivAssign::div_assign(&mut a2, b.clone());
            assert_eq!(a2, expected);
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_rem(
        a in strategies::uint(32),
        b in strategies::uint_nonzero(32),
    ) {
        if let Some(expected) = a.checked_rem(&b) {
            assert_eq!(Rem::rem(&a, &b), expected);
            assert_eq!(Rem::rem(a.clone(), &b), expected);
            assert_eq!(Rem::rem(&a, b.clone()), expected);
            assert_eq!(Rem::rem(a.clone(), b.clone()), expected);

            let mut a1 = a.clone();
            RemAssign::rem_assign(&mut a1, &b);
            assert_eq!(a1, expected);

            let mut a2 = a;
            RemAssign::rem_assign(&mut a2, b.clone());
            assert_eq!(a2, expected);
        }
    }
}

// Int ops

proptest! {
    #[test]
    fn prop_int_ops_sub(a in strategies::int(32), b in strategies::int(32)) {
        let expected = Sub::sub(a.clone(), &b);
        assert_eq!(Sub::sub(&a, &b), expected);
        assert_eq!(Sub::sub(a.clone(), b.clone()), expected);
        assert_eq!(Sub::sub(&a, b.clone()), expected);

        let mut a1 = a.clone();
        SubAssign::sub_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        SubAssign::sub_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_mul(a in strategies::int(16), b in strategies::int(16)) {
        let expected = Mul::mul(a.clone(), &b);
        assert_eq!(Mul::mul(&a, &b), expected);
        assert_eq!(Mul::mul(a.clone(), b.clone()), expected);
        assert_eq!(Mul::mul(&a, b.clone()), expected);

        let mut a1 = a.clone();
        MulAssign::mul_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        MulAssign::mul_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_bitand(a in strategies::int(32), b in strategies::int(32)) {
        let expected = BitAnd::bitand(a.clone(), &b);
        assert_eq!(BitAnd::bitand(&a, &b), expected);
        assert_eq!(BitAnd::bitand(a.clone(), b.clone()), expected);
        assert_eq!(BitAnd::bitand(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitAndAssign::bitand_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitAndAssign::bitand_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_bitor(a in strategies::int(32), b in strategies::int(32)) {
        let expected = BitOr::bitor(a.clone(), &b);
        assert_eq!(BitOr::bitor(&a, &b), expected);
        assert_eq!(BitOr::bitor(a.clone(), b.clone()), expected);
        assert_eq!(BitOr::bitor(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitOrAssign::bitor_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitOrAssign::bitor_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_bitxor(a in strategies::int(32), b in strategies::int(32)) {
        let expected = BitXor::bitxor(a.clone(), &b);
        assert_eq!(BitXor::bitxor(&a, &b), expected);
        assert_eq!(BitXor::bitxor(a.clone(), b.clone()), expected);
        assert_eq!(BitXor::bitxor(&a, b.clone()), expected);

        let mut a1 = a.clone();
        BitXorAssign::bitxor_assign(&mut a1, &b);
        assert_eq!(a1, expected);

        let mut a2 = a;
        BitXorAssign::bitxor_assign(&mut a2, b.clone());
        assert_eq!(a2, expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_div(
        a in strategies::int(32),
        b in strategies::int_nonzero(32),
    ) {
        if let Some(expected) = a.checked_div(&b) {
            assert_eq!(Div::div(&a, &b), expected);
            assert_eq!(Div::div(a.clone(), &b), expected);
            assert_eq!(Div::div(&a, b.clone()), expected);
            assert_eq!(Div::div(a.clone(), b.clone()), expected);

            let mut a1 = a.clone();
            DivAssign::div_assign(&mut a1, &b);
            assert_eq!(a1, expected);

            let mut a2 = a;
            DivAssign::div_assign(&mut a2, b.clone());
            assert_eq!(a2, expected);
        }
    }
}

proptest! {
    #[test]
    fn prop_int_ops_rem(
        a in strategies::int(32),
        b in strategies::int_nonzero(32),
    ) {
        if let Some(expected) = a.checked_rem(&b) {
            assert_eq!(Rem::rem(&a, &b), expected);
            assert_eq!(Rem::rem(a.clone(), &b), expected);
            assert_eq!(Rem::rem(&a, b.clone()), expected);
            assert_eq!(Rem::rem(a.clone(), b.clone()), expected);

            let mut a1 = a.clone();
            RemAssign::rem_assign(&mut a1, &b);
            assert_eq!(a1, expected);

            let mut a2 = a;
            RemAssign::rem_assign(&mut a2, b.clone());
            assert_eq!(a2, expected);
        }
    }
}

proptest! {
    #[test]
    fn prop_int_reference_div_rem_zero_rhs_panics(a in strategies::int(32)) {
        let div_owned_rhs_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(Div::div(&a, ArbiInt::zero()));
        }))
        .is_err();
        prop_assert!(div_owned_rhs_panicked);

        let div_borrowed_rhs_panicked = catch_unwind(AssertUnwindSafe(|| {
            let zero = ArbiInt::zero();
            drop(Div::div(&a, &zero));
        }))
        .is_err();
        prop_assert!(div_borrowed_rhs_panicked);

        let rem_owned_rhs_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(Rem::rem(&a, ArbiInt::zero()));
        }))
        .is_err();
        prop_assert!(rem_owned_rhs_panicked);

        let rem_borrowed_rhs_panicked = catch_unwind(AssertUnwindSafe(|| {
            let zero = ArbiInt::zero();
            drop(Rem::rem(&a, &zero));
        }))
        .is_err();
        prop_assert!(rem_borrowed_rhs_panicked);
    }
}

#[test]
fn division_assignment_validation_preserves_receiver() {
    let zero_unsigned = ArbiUint::zero();
    let unsigned_original = ArbiUint::from(37_u8);

    let mut unsigned_quotient = unsigned_original.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            DivAssign::div_assign(&mut unsigned_quotient, &zero_unsigned);
        }))
        .is_err()
    );
    assert_eq!(unsigned_quotient.value, unsigned_original.value);
    assert_eq!(unsigned_quotient.precision, unsigned_original.precision);

    let mut unsigned_remainder = unsigned_original.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            RemAssign::rem_assign(&mut unsigned_remainder, &zero_unsigned);
        }))
        .is_err()
    );
    assert_eq!(unsigned_remainder.value, unsigned_original.value);
    assert_eq!(unsigned_remainder.precision, unsigned_original.precision);

    let zero_signed = ArbiInt::zero();
    let signed_original = ArbiInt::from(-37_i8);
    let mut signed_zero_quotient = signed_original.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            DivAssign::div_assign(&mut signed_zero_quotient, &zero_signed);
        }))
        .is_err()
    );
    assert_eq!(signed_zero_quotient.value, signed_original.value);
    assert_eq!(signed_zero_quotient.precision, signed_original.precision);

    let mut signed_zero_remainder = signed_original.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            RemAssign::rem_assign(&mut signed_zero_remainder, &zero_signed);
        }))
        .is_err()
    );
    assert_eq!(signed_zero_remainder.value, signed_original.value);
    assert_eq!(signed_zero_remainder.precision, signed_original.precision);

    let width = BoundedPrecision::new(8).expect("eight is a valid bounded width");
    let minimum = ArbiInt::with_precision_checked(-128_i16, width)
        .expect("-128 is the signed eight-bit minimum");
    let minus_one =
        ArbiInt::with_precision_checked(-1_i8, width).expect("-1 fits eight signed bits");

    let mut minimum_quotient = minimum.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            DivAssign::div_assign(&mut minimum_quotient, &minus_one);
        }))
        .is_err()
    );
    assert_eq!(minimum_quotient.value, minimum.value);
    assert_eq!(minimum_quotient.precision, minimum.precision);

    let mut minimum_remainder = minimum.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            RemAssign::rem_assign(&mut minimum_remainder, &minus_one);
        }))
        .is_err()
    );
    assert_eq!(minimum_remainder.value, minimum.value);
    assert_eq!(minimum_remainder.precision, minimum.precision);
}

// Shifts

proptest! {
    #[test]
    fn prop_uint_ops_shl(a in strategies::uint(32), shift in 0_usize..=266) {
        let expected = Shl::shl(a.clone(), shift);
        assert_eq!(Shl::shl(&a, shift), expected);
        assert_eq!(Shl::shl(a, shift), expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_ops_shr(a in strategies::uint(32), shift in 0_usize..=266) {
        let expected = Shr::shr(a.clone(), shift);
        assert_eq!(Shr::shr(&a, shift), expected);
        assert_eq!(Shr::shr(a, shift), expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_shl(a in strategies::int(32), shift in 0_usize..=266) {
        let expected = Shl::shl(a.clone(), shift);
        assert_eq!(Shl::shl(&a, shift), expected);
        assert_eq!(Shl::shl(a, shift), expected);
    }
}

proptest! {
    #[test]
    fn prop_int_ops_shr(a in strategies::int(32), shift in 0_usize..=266) {
        let expected = Shr::shr(a.clone(), shift);
        assert_eq!(Shr::shr(&a, shift), expected);
        assert_eq!(Shr::shr(a, shift), expected);
    }
}
