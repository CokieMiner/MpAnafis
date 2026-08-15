//! Signed integer shift method APIs.

use core::ops::Shl;

use crate::error::MpError;

use super::{InternalMpInt, MpInt};

impl MpInt {
    // ------------------------------------------------------------------
    // Shift families (wrapping, overflowing, saturating, try)
    // ------------------------------------------------------------------

    /// Checked left shift. Returns `None` if the result exceeds the bounded precision.
    #[must_use]
    pub fn checked_shl(&self, shift: usize) -> Option<Self> {
        self.try_shl(shift).ok()
    }

    /// Wrapping left shift. Wraps the result within the bounded precision
    /// using two's complement wrapping.
    #[must_use]
    pub fn wrapping_shl(&self, shift: usize) -> Self {
        let value = self.precision.significant_bits().map_or_else(
            || Shl::shl(&self.value, shift),
            |bits| {
                if shift >= bits {
                    InternalMpInt::zero()
                } else {
                    Shl::shl(&self.value, shift).apply_wrapping(bits)
                }
            },
        );
        let result = Self {
            value,
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Overflowing left shift. Returns the result and a boolean indicating
    /// whether overflow occurred.
    #[must_use]
    pub fn overflowing_shl(&self, shift: usize) -> (Self, bool) {
        let (value, overflow) = self.precision.significant_bits().map_or_else(
            || (Shl::shl(&self.value, shift), false),
            |bits| {
                let overflow = self.value.bounded_shl_overflows(bits, shift);
                let value = if shift >= bits {
                    InternalMpInt::zero()
                } else {
                    Shl::shl(&self.value, shift).apply_wrapping(bits)
                };
                (value, overflow)
            },
        );
        let result = Self {
            value,
            precision: self.precision,
        };
        (result, overflow)
    }

    /// Saturating left shift. Saturates to the maximum (or minimum, for
    /// negative values) when overflow would occur.
    #[must_use]
    pub fn saturating_shl(&self, shift: usize) -> Self {
        if let Some(bits) = self.precision.significant_bits()
            && self.value.bounded_shl_overflows(bits, shift)
        {
            return if self.value.is_positive {
                Self::max_for_precision(bits)
            } else {
                Self::min_for_precision(bits)
            };
        }
        let result = Self {
            value: Shl::shl(&self.value, shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Tries to left shift.
    ///
    /// # Errors
    /// Returns `MpError::Overflow` if the result exceeds bounded
    /// precision.
    pub fn try_shl(&self, shift: usize) -> Result<Self, MpError> {
        if let Some(bits) = self.precision.significant_bits()
            && self.value.bounded_shl_overflows(bits, shift)
        {
            return Err(MpError::Overflow);
        }
        let result = Self {
            value: Shl::shl(&self.value, shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}
