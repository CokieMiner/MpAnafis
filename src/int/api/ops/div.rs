//! Division and remainder operator trait implementations.

use core::ops::{Div, DivAssign, Rem, RemAssign};

use super::{MpInt, MpUint};

// For unsigned `a` and non-zero `b`, `a / b <= a`; moreover, `a % b = a`
// when `a < b`, and otherwise `a % b < b <= a`. Therefore quotient and
// remainder fit the left operand's declared width. A non-assigning result uses
// a combined precision no narrower than the left operand, while assignment
// retains that left precision, so no post-result precision check is needed.
impl Div<Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn div(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.div_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Div<&Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn div(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.div_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Div<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn div(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = MpUint {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Div<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn div(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = MpUint {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl DivAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: Self) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl DivAssign<&Self> for MpUint {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &Self) {
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.div_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

impl Rem<Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn rem(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.rem_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Rem<&Self> for MpUint {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn rem(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.rem_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Rem<MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = MpUint {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Rem<&MpUint> for &MpUint {
    type Output = MpUint;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: &MpUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = MpUint {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl RemAssign<Self> for MpUint {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: Self) {
        RemAssign::rem_assign(self, &rhs);
    }
}

impl RemAssign<&Self> for MpUint {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: &Self) {
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.rem_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

// Signed truncating division has the same magnitude bounds: `|a / b| <= |a|`
// and `|a % b| <= |a|`. The sole quotient exception is bounded `MIN / -1`,
// which `assert_int_division_defined` rejects before any in-place mutation.
// Hence successful quotient and remainder results already fit the left (and
// therefore the combined) precision.
impl Div<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn div(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(&self, &rhs, precision.significant_bits(), "division");
        self.value.div_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Div<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn div(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(&self, rhs, precision.significant_bits(), "division");
        self.value.div_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Div<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn div(self, rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, &rhs, precision.significant_bits(), "division");
        let result = MpInt {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Div<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn div(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, rhs, precision.significant_bits(), "division");
        let result = MpInt {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl DivAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: Self) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl DivAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &Self) {
        assert_int_division_defined(self, rhs, self.precision.significant_bits(), "division");
        self.value.div_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

impl Rem<Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn rem(mut self, rhs: Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(&self, &rhs, precision.significant_bits(), "remainder");
        self.value.rem_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Rem<&Self> for MpInt {
    type Output = Self;
    #[inline]
    #[track_caller]
    fn rem(mut self, rhs: &Self) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(&self, rhs, precision.significant_bits(), "remainder");
        self.value.rem_assign(&rhs.value);
        self.precision = precision;
        self.debug_assert_valid();
        self
    }
}

impl Rem<MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, &rhs, precision.significant_bits(), "remainder");
        let result = MpInt {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Rem<&MpInt> for &MpInt {
    type Output = MpInt;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: &MpInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, rhs, precision.significant_bits(), "remainder");
        let result = MpInt {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl RemAssign<Self> for MpInt {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: Self) {
        RemAssign::rem_assign(self, &rhs);
    }
}

impl RemAssign<&Self> for MpInt {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: &Self) {
        assert_int_division_defined(self, rhs, self.precision.significant_bits(), "remainder");
        self.value.rem_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

#[inline]
#[track_caller]
fn assert_int_division_defined(
    lhs: &MpInt,
    rhs: &MpInt,
    destination_bits: Option<usize>,
    operation: &str,
) {
    assert!(!rhs.value.abs.is_zero(), "Division by zero");
    if let Some(bits) = destination_bits {
        assert!(
            !lhs.value.bounded_division_overflows(&rhs.value, bits),
            "MpInt {operation} overflow for Bounded({bits})"
        );
    }
}
