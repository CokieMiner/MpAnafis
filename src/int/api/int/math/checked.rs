//! Checked, result-based, and strict signed arithmetic APIs.

use crate::error::ArbiError;

use super::ArbiInt;

impl ArbiInt {
    // checked_* arithmetic
    // ------------------------------------------------------------------

    /// Checked addition. Returns `None` if the result exceeds the bounded
    /// precision.
    #[must_use]
    pub fn checked_add(&self, rhs: &Self) -> Option<Self> {
        self.try_add(rhs).ok()
    }

    /// Checked subtraction. Returns `None` if the result exceeds the bounded
    /// precision.
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

    /// Checked division. Returns `None` if the divisor is zero or if the
    /// quotient exceeds the bounded precision, including `MIN / -1`.
    #[must_use]
    pub fn checked_div(&self, rhs: &Self) -> Option<Self> {
        self.try_div(rhs).ok()
    }

    /// Checked remainder. Returns `None` if the divisor is zero or if the
    /// operation is the bounded signed `MIN / -1` overflow case.
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
    /// Returns `ArbiError::Overflow` if the result exceeds precision bounds.
    pub fn try_add(&self, rhs: &Self) -> Result<Self, ArbiError> {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let sum = self.value.add(&rhs.value);
        let result = Self {
            value: sum,
            precision: p,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return Err(ArbiError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to subtract two values.
    ///
    /// # Errors
    /// Returns `ArbiError::Overflow` if the result exceeds precision bounds.
    pub fn try_sub(&self, rhs: &Self) -> Result<Self, ArbiError> {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let diff = self.value.sub(&rhs.value);
        let result = Self {
            value: diff,
            precision: p,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return Err(ArbiError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to multiply two values.
    ///
    /// # Errors
    /// Returns `ArbiError::Overflow` if the result exceeds precision bounds.
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, ArbiError> {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let prod = self.value.mul(&rhs.value);
        let result = Self {
            value: prod,
            precision: p,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return Err(ArbiError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to divide two values.
    ///
    /// # Errors
    /// Returns `ArbiError::DivisionByZero` if `rhs` is zero, or
    /// `ArbiError::Overflow` if the quotient exceeds precision bounds,
    /// including `MIN / -1`.
    pub fn try_div(&self, rhs: &Self) -> Result<Self, ArbiError> {
        if rhs.value.abs.is_zero() {
            return Err(ArbiError::DivisionByZero);
        }
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if let Some(bits) = p.significant_bits()
            && self.value.bounded_division_overflows(&rhs.value, bits)
        {
            return Err(ArbiError::Overflow);
        }
        let quot = self.value.div(&rhs.value);
        let result = Self {
            value: quot,
            precision: p,
        };
        // Except for the rejected `MIN / -1` endpoint, truncating division
        // satisfies `|self / rhs| <= |self|`, so the quotient already fits.
        result.debug_assert_valid();
        Ok(result)
    }

    /// Tries to compute remainder.
    ///
    /// # Errors
    /// Returns `ArbiError::DivisionByZero` if `rhs` is zero, or
    /// `ArbiError::Overflow` for the bounded signed `MIN / -1` case.
    pub fn try_rem(&self, rhs: &Self) -> Result<Self, ArbiError> {
        if rhs.value.abs.is_zero() {
            return Err(ArbiError::DivisionByZero);
        }
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if let Some(bits) = p.significant_bits()
            && self.value.bounded_division_overflows(&rhs.value, bits)
        {
            return Err(ArbiError::Overflow);
        }
        let rem = self.value.rem(&rhs.value);
        let result = Self {
            value: rem,
            precision: p,
        };
        // `|self % rhs| <= |self|`, so a valid remainder already fits the
        // combined precision.
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
    /// Panics if the result exceeds the bounded signed precision.
    #[must_use]
    pub fn strict_sub(&self, rhs: &Self) -> Self {
        self.checked_sub(rhs)
            .expect("strict_sub: result exceeds precision")
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
    /// Panics if division by zero or the quotient exceeds the bounded
    /// precision.
    #[must_use]
    pub fn strict_div(&self, rhs: &Self) -> Self {
        self.checked_div(rhs)
            .expect("strict_div: division by zero or result exceeds precision")
    }

    /// Strict remainder.
    ///
    /// # Panics
    /// Panics if division by zero or the remainder exceeds the bounded
    /// precision.
    #[must_use]
    pub fn strict_rem(&self, rhs: &Self) -> Self {
        self.checked_rem(rhs)
            .expect("strict_rem: division by zero or result exceeds precision")
    }
}
