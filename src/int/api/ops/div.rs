//! Division and remainder operator trait implementations.

use core::ops::{Div, DivAssign, Rem, RemAssign};

use super::{ArbiInt, ArbiUint};

// For unsigned `a` and non-zero `b`, `a / b <= a`; moreover, `a % b = a`
// when `a < b`, and otherwise `a % b < b <= a`. Therefore quotient and
// remainder fit the left operand's declared width. A non-assigning result uses
// a combined precision no narrower than the left operand, while assignment
// retains that left precision, so no post-result precision check is needed.
impl Div<Self> for ArbiUint {
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

impl Div<&Self> for ArbiUint {
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

impl Div<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn div(self, rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = ArbiUint {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Div<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn div(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = ArbiUint {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl DivAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: Self) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl DivAssign<&Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &Self) {
        assert!(!rhs.value.is_zero(), "Division by zero");
        self.value.div_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

impl Rem<Self> for ArbiUint {
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

impl Rem<&Self> for ArbiUint {
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

impl Rem<ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = ArbiUint {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Rem<&ArbiUint> for &ArbiUint {
    type Output = ArbiUint;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: &ArbiUint) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert!(!rhs.value.is_zero(), "Division by zero");
        let result = ArbiUint {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl RemAssign<Self> for ArbiUint {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: Self) {
        RemAssign::rem_assign(self, &rhs);
    }
}

impl RemAssign<&Self> for ArbiUint {
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
impl Div<Self> for ArbiInt {
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

impl Div<&Self> for ArbiInt {
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

impl Div<ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn div(self, rhs: ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, &rhs, precision.significant_bits(), "division");
        let result = ArbiInt {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Div<&ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn div(self, rhs: &ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, rhs, precision.significant_bits(), "division");
        let result = ArbiInt {
            value: self.value.div(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl DivAssign<Self> for ArbiInt {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: Self) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl DivAssign<&Self> for ArbiInt {
    #[inline]
    #[track_caller]
    fn div_assign(&mut self, rhs: &Self) {
        assert_int_division_defined(self, rhs, self.precision.significant_bits(), "division");
        self.value.div_assign(&rhs.value);
        self.debug_assert_valid();
    }
}

impl Rem<Self> for ArbiInt {
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

impl Rem<&Self> for ArbiInt {
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

impl Rem<ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, &rhs, precision.significant_bits(), "remainder");
        let result = ArbiInt {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl Rem<&ArbiInt> for &ArbiInt {
    type Output = ArbiInt;
    #[inline]
    #[track_caller]
    fn rem(self, rhs: &ArbiInt) -> Self::Output {
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        assert_int_division_defined(self, rhs, precision.significant_bits(), "remainder");
        let result = ArbiInt {
            value: self.value.rem(&rhs.value),
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

impl RemAssign<Self> for ArbiInt {
    #[inline]
    #[track_caller]
    fn rem_assign(&mut self, rhs: Self) {
        RemAssign::rem_assign(self, &rhs);
    }
}

impl RemAssign<&Self> for ArbiInt {
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
    lhs: &ArbiInt,
    rhs: &ArbiInt,
    destination_bits: Option<usize>,
    operation: &str,
) {
    assert!(!rhs.value.abs.is_zero(), "Division by zero");
    if let Some(bits) = destination_bits {
        assert!(
            !lhs.value.bounded_division_overflows(&rhs.value, bits),
            "ArbiInt {operation} overflow for Bounded({bits})"
        );
    }
}
