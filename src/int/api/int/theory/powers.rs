//! Exponentiation APIs for signed integers.

use crate::error::ArbiError;

use super::{ArbiInt, InternalArbiInt, InternalArbiUint};

impl ArbiInt {
    /// Returns `self` raised to the power of `exp`.
    ///
    /// # Panics
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn pow(&self, exp: u32) -> Self {
        self.try_pow(exp).expect("pow exceeds bounded precision")
    }

    /// Returns `self` raised to the power of `exp`, or `None` if the result
    /// exceeds the bounded precision.
    #[must_use]
    pub fn checked_pow(&self, exp: u32) -> Option<Self> {
        self.try_pow(exp).ok()
    }

    /// Tries to raise `self` to the power of `exp`.
    ///
    /// # Errors
    /// Returns `ArbiError::Overflow` if the result exceeds precision bounds.
    pub fn try_pow(&self, exp: u32) -> Result<Self, ArbiError> {
        if exp == 0 {
            let result = Self {
                value: InternalArbiInt {
                    abs: InternalArbiUint::one(),
                    is_positive: true,
                },
                precision: self.precision,
            };
            if let Some(bits) = result.precision.significant_bits()
                && result.value.required_signed_bits_for_bounded_storage() > bits
            {
                return Err(ArbiError::Overflow);
            }
            result.debug_assert_valid();
            return Ok(result);
        }
        if self.is_zero() {
            let result = Self {
                value: InternalArbiInt::zero(),
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Ok(result);
        }
        if self.value.abs.is_one() {
            let is_positive = !self.is_negative() || exp.is_multiple_of(2);
            let result = Self {
                value: InternalArbiInt {
                    abs: self.value.abs.clone(),
                    is_positive,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Ok(result);
        }
        let prod = self.value.abs.pow(exp);
        let is_positive = !self.is_negative() || exp.is_multiple_of(2);
        let result = Self {
            value: InternalArbiInt {
                abs: prod,
                is_positive,
            },
            precision: self.precision,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return Err(ArbiError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }
}
