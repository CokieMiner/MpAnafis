//! Unsigned integer bitwise inspection and manipulation APIs.
//!
//! Shift policies live in [`shift`](super::shift), mirroring the signed side.

use crate::error::MpError;

use super::{MpUint, Precision};

impl MpUint {
    // Bitwise
    /// Finds the index of the first zero bit.
    #[must_use]
    pub fn find_first_zero_bit(&self) -> usize {
        self.value.find_first_zero_bit()
    }

    // Bit manipulation
    // ------------------------------------------------------------------

    /// Rotates the bits left by `n` positions within the given `width`.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_left(&self, n: u32, width: usize) -> Option<Self> {
        let precision = Precision::new_bounded(width)?;
        let result = Self {
            value: self.value.rotate_left(n, width),
            precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Rotates the bits right by `n` positions within the given `width`.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_right(&self, n: u32, width: usize) -> Option<Self> {
        let precision = Precision::new_bounded(width)?;
        let result = Self {
            value: self.value.rotate_right(n, width),
            precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Reverses the bits within the given `width`.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn reverse_bits(&self, width: usize) -> Option<Self> {
        let precision = Precision::new_bounded(width)?;
        let result = Self {
            value: self.value.reverse_bits(width),
            precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Swaps all bytes in the representation.
    #[must_use]
    pub fn swap_bytes(&self) -> Self {
        let result = Self {
            value: self.value.swap_bytes(self.precision.significant_bits()),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Computes the bitwise NOT within the given `width`.
    ///
    /// The complement of an unsigned value is only defined against a width, so
    /// there is no unlimited form: `!self` panics on an unlimited `MpUint`
    /// and this method is what a caller reaches for instead.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn not_with_width(&self, width: usize) -> Option<Self> {
        let precision = Precision::new_bounded(width)?;
        let result = Self {
            value: self.value.not(width),
            precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Computes the bitwise NOT within this value's own bounded precision.
    ///
    /// # Errors
    /// Returns `MpError::WidthRequired` when the precision is unlimited,
    /// because the complement has no meaning without a width. Use
    /// [`MpUint::not_with_width`] to supply one explicitly.
    pub fn try_not(&self) -> Result<Self, MpError> {
        let width = self
            .precision
            .significant_bits()
            .ok_or(MpError::WidthRequired)?;
        self.not_with_width(width).ok_or(MpError::WidthRequired)
    }

    /// Returns the number of leading zeros.
    /// Returns `None` for unlimited precision (width-dependent operation).
    #[must_use]
    pub fn leading_zeros(&self) -> Option<usize> {
        self.precision
            .significant_bits()
            .map(|width| self.value.leading_zeros_for_width(width))
    }

    /// Returns the number of leading ones.
    /// Returns `None` for unlimited precision (width-dependent operation).
    #[must_use]
    pub fn leading_ones(&self) -> Option<usize> {
        self.precision
            .significant_bits()
            .map(|width| self.value.leading_ones_for_width(width))
    }

    /// Returns the number of trailing zeros in the binary representation.
    #[must_use]
    pub fn trailing_zeros(&self) -> usize {
        self.value.trailing_zeros()
    }

    /// Returns the number of trailing ones.
    #[must_use]
    pub fn trailing_ones(&self) -> usize {
        self.value.trailing_ones()
    }

    /// Returns the number of ones in the binary representation.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.value.count_ones()
    }

    /// Returns the number of zeros in the binary representation.
    /// Returns `None` for unlimited precision (width-dependent operation).
    #[must_use]
    pub fn count_zeros(&self) -> Option<usize> {
        self.precision
            .significant_bits()
            .map(|bits| self.value.count_zeros_for_width(bits))
    }

    /// Returns the value of the bit at position `bit` (0-indexed).
    #[must_use]
    pub fn get_bit(&self, bit: usize) -> bool {
        self.value.get_bit(bit)
    }

    /// Sets the bit at `bit` to the given `value` and returns a new value.
    #[must_use]
    pub fn set_bit_to(&self, bit: usize, value: bool) -> Self {
        if let Some(width) = self.precision.significant_bits()
            && bit >= width
        {
            return self.clone();
        }
        let result = Self {
            value: self.value.set_bit_to(bit, value),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Returns `true` if the bit at position `bit` is set (alias for `get_bit`).
    #[must_use]
    pub fn test_bit(&self, bit: usize) -> bool {
        self.get_bit(bit)
    }

    /// Sets the bit at position `bit` to `1` and returns a new value.
    #[must_use]
    pub fn set_bit(&self, bit: usize) -> Self {
        self.set_bit_to(bit, true)
    }

    /// Clears the bit at position `bit` (sets to `0`) and returns a new value.
    #[must_use]
    pub fn clear_bit(&self, bit: usize) -> Self {
        self.set_bit_to(bit, false)
    }

    /// Toggles (flips) the bit at position `bit` and returns a new value.
    #[must_use]
    pub fn toggle_bit(&self, bit: usize) -> Self {
        self.set_bit_to(bit, !self.get_bit(bit))
    }

    /// Finds the index of the first (least significant) set bit.
    #[must_use]
    pub fn find_first_set_bit(&self) -> Option<usize> {
        self.value.find_first_set_bit()
    }

    /// Finds the index of the next set bit after `from`.
    #[must_use]
    pub fn find_next_set_bit(&self, from: usize) -> Option<usize> {
        self.value.find_next_set_bit(from)
    }

    /// Finds the index of the next zero bit after `from`.
    #[must_use]
    pub fn find_next_zero_bit(&self, from: usize) -> usize {
        self.value.find_next_zero_bit(from)
    }

    /// Extracts a range of bits `[from, to)` as a new value.
    #[must_use]
    pub fn bit_range(&self, from: usize, to: usize) -> Self {
        let result = Self {
            value: self.value.bit_range(from, to),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    // ------------------------------------------------------------------
}
