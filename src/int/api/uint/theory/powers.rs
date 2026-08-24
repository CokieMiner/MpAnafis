//! Exponentiation APIs for unsigned integers.

use crate::error::MpError;

use super::{InternalMpUint, MpUint};

impl MpUint {
    /// Returns `self` raised to the power of `exp`.
    ///
    /// # Panics
    ///
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn pow(&self, exp: u32) -> Self {
        self.try_pow(exp).expect("pow exceeds bounded precision")
    }

    /// Returns `self` raised to the power of `exp`, or `None` if the result overflows
    /// the bounded precision.
    #[must_use]
    pub fn checked_pow(&self, exp: u32) -> Option<Self> {
        self.try_pow(exp).ok()
    }

    /// Tries to raise `self` to the power of `exp`.
    ///
    /// # Errors
    /// Returns `MpError::Overflow` if the result exceeds precision bounds.
    pub fn try_pow(&self, exp: u32) -> Result<Self, MpError> {
        if exp == 0 {
            let result = Self {
                value: InternalMpUint::one(),
                precision: self.precision,
            };
            // `one()` needs a single bit and every bounded precision is at
            // least one bit, so the result provably fits: no overflow check.
            result.debug_assert_valid();
            return Ok(result);
        }
        if self.is_zero() || self.value.is_one() {
            let result = Self {
                value: self.value.clone(),
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Ok(result);
        }
        let prod = self.value.pow(exp);
        let result = Self {
            value: prod,
            precision: self.precision,
        };
        if let Some(bits) = result.precision.significant_bits()
            && result.value.significant_bits() > bits
        {
            return Err(MpError::Overflow);
        }
        result.debug_assert_valid();
        Ok(result)
    }

    /// Returns `true` if `self` is a power of two.
    #[must_use]
    pub fn is_power_of_two(&self) -> bool {
        self.value.is_power_of_two()
    }

    /// Returns the smallest power of two greater than or equal to `self`, or `None` if the result overflows.
    #[must_use]
    pub fn checked_next_power_of_two(&self) -> Option<Self> {
        if self.value.is_zero() {
            let result = Self {
                value: InternalMpUint::one(),
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Some(result);
        }
        if self.value.is_power_of_two() {
            return Some(self.clone());
        }
        let next_bit = self.value.significant_bits();
        if let Some(width) = self.precision.significant_bits()
            && next_bit >= width
        {
            return None;
        }
        let result = Self {
            value: InternalMpUint::one().shl(next_bit),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }
}
