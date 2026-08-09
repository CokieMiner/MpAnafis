//! Core signed integer representation and two's-complement conversion helpers.

#![allow(
    unsafe_code,
    reason = "Using unwrap_unchecked where width > 0 is guaranteed by caller or check, avoiding panic branch in compiled binary."
)]
use super::InternalMpUint;

/// The core signed arbitrary precision integer.
#[derive(Debug, Clone)]
pub struct InternalMpInt {
    /// The unsigned magnitude of the integer.
    pub(crate) abs: InternalMpUint,
    /// Sign flag: true if positive or zero, false if strictly negative.
    pub(crate) is_positive: bool,
}

impl InternalMpInt {
    /// Creates a new arbitrary precision integer with the value 0.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            abs: InternalMpUint::zero(),
            is_positive: true,
        }
    }

    /// Creates a new arbitrary precision integer with the value 1.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self {
            abs: InternalMpUint::one(),
            is_positive: true,
        }
    }

    /// Pre-allocates memory for a specific number of limbs.
    #[inline]
    #[must_use]
    pub fn with_capacity(limbs: usize) -> Self {
        Self {
            abs: InternalMpUint::with_capacity(limbs),
            is_positive: true,
        }
    }

    /// Returns the required bit width for bounded storage of this signed magnitude.
    #[must_use]
    pub fn required_signed_bits_for_bounded_storage(&self) -> usize {
        let sig = self.abs.significant_bits();
        if self.is_positive {
            // positive values need a sign bit (0)
            sig.wrapping_add(1)
        } else {
            // negative values need at least sig + 1 bits,
            // EXCEPT when magnitude is exactly a power of 2
            if self.abs.is_power_of_two() {
                sig
            } else {
                sig.wrapping_add(1)
            }
        }
    }

    /// Returns whether shifting this value left exceeds a caller-proved
    /// bounded signed width.
    #[inline]
    #[must_use]
    pub fn bounded_shl_overflows(&self, bits: usize, shift: usize) -> bool {
        let value_bits = self.required_signed_bits_for_bounded_storage();
        debug_assert!(
            value_bits <= bits,
            "bounded signed value must fit its declared precision"
        );
        // For non-zero x, signed storage requires `value_bits + shift` bits.
        // The caller proves `value_bits <= bits`, so the wrapping subtraction
        // equals the exact non-negative slack and avoids a second checked path.
        !self.abs.is_zero() && shift > bits.wrapping_sub(value_bits)
    }

    /// Returns `true` if this integer is the minimum signed value for the given
    /// bit width (i.e., `-2^(width_bits - 1)`).
    #[must_use]
    pub fn is_signed_min_for_width(&self, width_bits: usize) -> bool {
        if self.is_positive || self.abs.is_zero() {
            return false;
        }
        let sig = self.abs.significant_bits();
        sig == width_bits && self.abs.is_power_of_two()
    }

    /// Converts this signed integer into a caller-validated non-zero two's-complement width.
    #[must_use]
    pub fn to_tc_bits(&self, width: usize) -> InternalMpUint {
        debug_assert!(width > 0, "two's-complement width must be non-zero");
        if self.is_positive {
            self.abs.clone().apply_wrapping(width)
        } else {
            self.abs.clone().apply_negate_wrapping(width)
        }
    }

    /// Constructs an `InternalMpInt` from a two's-complement unsigned integer of `width` bits.
    #[must_use]
    pub fn from_tc_bits(bits: InternalMpUint, width: usize) -> Self {
        debug_assert!(width > 0, "two's-complement width must be non-zero");
        let wrapped_bits = bits.apply_wrapping(width);
        if wrapped_bits.is_zero() {
            return Self::zero();
        }
        if wrapped_bits.get_bit(width.saturating_sub(1)) {
            let mut abs = wrapped_bits.not(width);
            abs.increment();
            Self {
                abs,
                is_positive: false,
            }
        } else {
            Self {
                abs: wrapped_bits,
                is_positive: true,
            }
        }
    }

    /// Normalizes the signed representation so zero is always positive.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        if self.abs.is_zero() {
            self.is_positive = true;
        }
        self
    }
}

#[cfg(test)]
#[path = "tests/mpint.rs"]
mod tests;
