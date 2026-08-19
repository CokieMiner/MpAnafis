//! Checked, result-based, and strict unsigned arithmetic APIs.

use crate::error::MpError;

use super::MpUint;

impl MpUint {
    // checked_* arithmetic
    // ------------------------------------------------------------------

    /// Checked addition. Returns `None` if the result exceeds the bounded
    /// precision or if the operands' precisions cannot hold the result.
    #[must_use]
    pub fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.try_add(rhs).ok()
    }

    /// Checked subtraction. Returns `None` if `self < rhs`.
    #[must_use]
    pub fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        self.try_sub(rhs).ok()
    }

    /// Checked multiplication. Returns `None` if the result exceeds the
    /// bounded precision.
    #[must_use]
    pub fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        self.try_mul(rhs).ok()
    }

    /// Checked division. Returns `None` if the divisor is zero.
    #[must_use]
    pub fn checked_div(&self, rhs: &Self) -> Option<Self> {
        self.try_div(rhs).ok()
    }

    /// Checked remainder. Returns `None` if the divisor is zero.
    #[must_use]
    pub fn checked_rem(&self, rhs: &Self) -> Option<Self> {
        self.try_rem(rhs).ok()
    }

    // ------------------------------------------------------------------
    // try_* arithmetic (Result-based)
    // ------------------------------------------------------------------

    /// Tries to add two values.
    ///
    /// # Errors
    /// Returns `MpError::Overflow` if the result exceeds precision bounds.
    pub fn try_add(&self, rhs: &Self) -> Result<Self, MpError> {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let sum = self.value.add(&rhs.value);
        let result = Self {
            value: sum,
            precision: p,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.significant_bits() > bits
        {
            return Err(MpError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to subtract two values.
    ///
    /// # Errors
    /// Returns `MpError::Underflow` if `self < rhs`.
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, MpError> {
        let (diff, underflowed) = self.value.sub_with_underflow(&rhs.value);
        if underflowed {
            return Err(MpError::Underflow);
        }
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: diff,
            precision: p,
        };
        // `self - rhs <= self`, and the combined precision is at least the
        // left precision, so a successful unsigned subtraction always fits.
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to multiply two values.
    ///
    /// # Errors
    /// Returns `MpError::Overflow` if the result exceeds precision bounds.
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, MpError> {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let prod = self.value.mul(&rhs.value);
        let result = Self {
            value: prod,
            precision: p,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.significant_bits() > bits
        {
            return Err(MpError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to divide two values.
    ///
    /// # Errors
    /// Returns `MpError::DivisionByZero` if `rhs` is zero.
    pub fn try_div(&self, rhs: &Self) -> Result<Self, MpError> {
        if rhs.value.is_zero() {
            return Err(MpError::DivisionByZero);
        }
        let quot = self.value.div(&rhs.value);
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: quot,
            precision: p,
        };
        // For non-zero `rhs`, `self / rhs <= self`; the combined precision is
        // at least the left precision, so the quotient always fits.
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to compute remainder.
    ///
    /// # Errors
    /// Returns `MpError::DivisionByZero` if `rhs` is zero.
    pub fn try_rem(&self, rhs: &Self) -> Result<Self, MpError> {
        if rhs.value.is_zero() {
            return Err(MpError::DivisionByZero);
        }
        let rem = self.value.rem(&rhs.value);
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: rem,
            precision: p,
        };
        // A valid remainder is at most `self`; the combined precision is at
        // least the left precision, so it always fits.
        result.debug_assert_valid();
        Ok(result)
    }

    // ------------------------------------------------------------------
    // strict_* arithmetic
    // ------------------------------------------------------------------

    /// Strict addition.
    ///
    /// # Panics
    /// Panics if the result exceeds the bounded precision.
    #[must_use]
    pub fn strict_add(&self, rhs: &Self) -> Self {
        self.checked_add(rhs)
            .expect("strict_add: result exceeds precision")
    }

    /// Strict subtraction.
    ///
    /// # Panics
    /// Panics if `rhs` is greater than `self`.
    #[must_use]
    pub fn strict_sub(&self, rhs: &Self) -> Self {
        self.checked_sub(rhs)
            .expect("strict_sub: unsigned underflow")
    }

    /// Strict multiplication.
    ///
    /// # Panics
    /// Panics if the result exceeds the bounded precision.
    #[must_use]
    pub fn strict_mul(&self, rhs: &Self) -> Self {
        self.checked_mul(rhs)
            .expect("strict_mul: result exceeds precision")
    }

    /// Strict division.
    ///
    /// # Panics
    /// Panics if division by zero.
    #[must_use]
    pub fn strict_div(&self, rhs: &Self) -> Self {
        self.checked_div(rhs).expect("strict_div: division by zero")
    }

    /// Strict remainder.
    ///
    /// # Panics
    /// Panics if division by zero.
    #[must_use]
    pub fn strict_rem(&self, rhs: &Self) -> Self {
        self.checked_rem(rhs).expect("strict_rem: division by zero")
    }
}
