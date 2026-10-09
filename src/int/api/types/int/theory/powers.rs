//! Exponentiation APIs for signed integers.

use crate::error::MpError;

use super::{InternalMpInt, InternalMpUint, MpInt};

impl MpInt {
    /// Computes the square `self * self`.
    ///
    /// # Panics
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn square(&self) -> Self {
        let result = Self {
            value: self.value.square(),
            precision: self.precision,
        };
        result.assert_fits("square");
        result.debug_assert_valid();
        result
    }

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
    /// Returns `MpError::Overflow` if the result exceeds precision bounds.
    pub fn try_pow(&self, exp: u32) -> Result<Self, MpError> {
        if exp == 1 {
            return Ok(self.clone());
        }
        if exp == 0 {
            let result = Self {
                value: InternalMpInt {
                    abs: InternalMpUint::one(),
                    is_positive: true,
                },
                precision: self.precision,
            };
            if let Some(bits) = result.precision.significant_bits()
                && result.value.required_signed_bits_for_bounded_storage() > bits
            {
                return Err(MpError::Overflow);
            }
            result.debug_assert_valid();
            return Ok(result);
        }
        if self.is_zero() {
            let result = Self {
                value: InternalMpInt::zero(),
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Ok(result);
        }
        if self.value.abs.is_one() {
            let is_positive = !self.is_negative() || exp.is_multiple_of(2);
            // A signed one-bit width represents only -1 and 0.
            if is_positive && self.precision.significant_bits() == Some(1) {
                return Err(MpError::Overflow);
            }
            let result = Self {
                value: InternalMpInt {
                    abs: self.value.abs.clone(),
                    is_positive,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Ok(result);
        }
        // The preceding cases establish exp > 1 and a magnitude greater than one.
        let prod = self.value.abs.pow_nontrivial(exp);
        let is_positive = !self.is_negative() || exp.is_multiple_of(2);
        let result = Self {
            value: InternalMpInt {
                abs: prod,
                is_positive,
            },
            precision: self.precision,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return Err(MpError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }
}
