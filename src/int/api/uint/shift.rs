//! Unsigned shift policies.
//!
//! Only the left shift has a policy family: shifting right can never leave a
//! bounded precision, so `Shr` alone covers it. Each policy differs solely in
//! what it does when bits leave the declared width, which under
//! [`Precision::Unlimited`](crate::Precision::Unlimited) never happens —
//! the width check short-circuits and all five degenerate to the same shift.

use crate::error::MpError;

use super::{InternalMpUint, MpUint};

impl MpUint {
    /// Checked left shift. Returns `None` if the result overflows bounded precision.
    #[must_use]
    pub fn checked_shl(&self, shift: usize) -> Option<Self> {
        if let Some(bits) = self.precision.significant_bits()
            && self.value.bounded_shl_overflows(bits, shift)
        {
            return None;
        }
        let result = Self {
            value: self.value.shl(shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Wrapping left shift. Wraps the result within the bounded precision.
    #[must_use]
    pub fn wrapping_shl(&self, shift: usize) -> Self {
        let value = self.precision.significant_bits().map_or_else(
            || self.value.shl(shift),
            |bits| {
                if shift >= bits {
                    InternalMpUint::zero()
                } else {
                    self.value.shl(shift).apply_wrapping(bits)
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
    /// whether overflow occurred (bits shifted out of bounded precision).
    #[must_use]
    pub fn overflowing_shl(&self, shift: usize) -> (Self, bool) {
        let (value, overflow) = self.precision.significant_bits().map_or_else(
            || (self.value.shl(shift), false),
            |bits| {
                let overflow = self.value.bounded_shl_overflows(bits, shift);
                let value = if shift >= bits {
                    InternalMpUint::zero()
                } else {
                    self.value.shl(shift).apply_wrapping(bits)
                };
                (value, overflow)
            },
        );
        let result = Self {
            value,
            precision: self.precision,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    /// Saturating left shift. Saturates to `max_for_precision` if any
    /// bits are shifted out of bounded precision.
    #[must_use]
    pub fn saturating_shl(&self, shift: usize) -> Self {
        let value = self.precision.significant_bits().map_or_else(
            || self.value.shl(shift),
            |bits| {
                if self.value.bounded_shl_overflows(bits, shift) {
                    InternalMpUint::max_for_bits(bits)
                } else {
                    self.value.shl(shift)
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

    /// Tries to left shift.
    ///
    /// # Errors
    /// Returns `MpError::Overflow` if the shift exceeds bounded
    /// precision.
    pub fn try_shl(&self, shift: usize) -> Result<Self, MpError> {
        if let Some(bits) = self.precision.significant_bits()
            && self.value.bounded_shl_overflows(bits, shift)
        {
            return Err(MpError::Overflow);
        }
        let result = Self {
            value: self.value.shl(shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}
