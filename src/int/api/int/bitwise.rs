//! Signed integer bitwise inspection and manipulation APIs.

use core::cmp::Ordering;

use crate::error::MpError;

use super::{InternalMpInt, InternalMpUint, MpInt, Precision};

impl MpInt {
    fn to_tc_bits(&self, width: usize) -> Option<InternalMpUint> {
        let _precision = Precision::new_bounded(width)?;
        Some(self.value.to_tc_bits(width))
    }

    fn from_tc_bits_with_width(bits: InternalMpUint, width: usize) -> Option<Self> {
        let precision = Precision::new_bounded(width)?;
        let internal = InternalMpInt::from_tc_bits(bits, width);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Rotates the value left within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_left(&self, n: u32, width: usize) -> Option<Self> {
        let bits = self.to_tc_bits(width)?;
        let rotated = bits.rotate_left(n, width);
        Self::from_tc_bits_with_width(rotated, width)
    }

    /// Rotates the value right within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_right(&self, n: u32, width: usize) -> Option<Self> {
        let bits = self.to_tc_bits(width)?;
        let rotated = bits.rotate_right(n, width);
        Self::from_tc_bits_with_width(rotated, width)
    }

    /// Reverses the bits within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn reverse_bits(&self, width: usize) -> Option<Self> {
        let bits = self.to_tc_bits(width)?;
        let reversed = bits.reverse_bits(width);
        Self::from_tc_bits_with_width(reversed, width)
    }

    /// Computes the bitwise NOT within the given width.
    ///
    /// `!self` is already total on `MpInt`, including at unlimited precision,
    /// because the sign supplies the infinite extension. This method exists for
    /// the cases where the caller wants the complement of a *specific* width
    /// rather than of the value.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn not_with_width(&self, width: usize) -> Option<Self> {
        let bits = self.to_tc_bits(width)?;
        let complement = bits.not(width);
        Self::from_tc_bits_with_width(complement, width)
    }

    /// Computes the bitwise NOT within this value's own bounded precision.
    ///
    /// # Errors
    /// Returns `MpError::WidthRequired` when the precision is unlimited. Note
    /// that `!self` is still defined there and is the operation to reach for;
    /// this method is the fallible width-bound form.
    pub fn try_not(&self) -> Result<Self, MpError> {
        let width = self
            .precision
            .significant_bits()
            .ok_or(MpError::WidthRequired)?;
        self.not_with_width(width).ok_or(MpError::WidthRequired)
    }

    /// Returns the number of leading zeros within the current bounded width.
    #[must_use]
    pub fn leading_zeros(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        let bits = self.to_tc_bits(width)?;
        Some(bits.leading_zeros_for_width(width))
    }

    /// Returns the number of leading ones within the current bounded width.
    #[must_use]
    pub fn leading_ones(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        let bits = self.to_tc_bits(width)?;
        Some(bits.leading_ones_for_width(width))
    }

    /// Returns the number of zero bits within the current bounded width.
    #[must_use]
    pub fn count_zeros(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        let bits = self.to_tc_bits(width)?;
        Some(bits.count_zeros_for_width(width))
    }

    /// Returns the value of the bit at position `bit`.
    #[must_use]
    pub fn get_bit(&self, bit: usize) -> bool {
        if let Some(width) = self.precision.significant_bits()
            && bit >= width
        {
            return false;
        }
        if self.is_negative() {
            let tz = self.value.abs.trailing_zeros();
            match bit.cmp(&tz) {
                Ordering::Less => false,
                Ordering::Equal => true,
                Ordering::Greater => !self.value.abs.get_bit(bit),
            }
        } else {
            self.value.abs.get_bit(bit)
        }
    }

    /// Sets the bit at `bit` to the given `value`.
    #[must_use]
    pub fn set_bit_to(&self, bit: usize, value: bool) -> Self {
        if let Some(width) = self.precision.significant_bits() {
            if bit >= width {
                return self.clone();
            }
            if let Some(mut bits) = self.to_tc_bits(width) {
                bits = bits.set_bit_to(bit, value);
                if let Some(result) = Self::from_tc_bits_with_width(bits, width) {
                    return result;
                }
            }
        }
        if self.is_negative() {
            if self.get_bit(bit) == value {
                self.clone()
            } else {
                let bit_val = Self {
                    value: InternalMpInt {
                        abs: InternalMpUint::one().shl(bit),
                        is_positive: true,
                    },
                    precision: self.precision,
                };
                let internal = if value {
                    self.value.add(&bit_val.value)
                } else {
                    self.value.sub(&bit_val.value)
                };
                let result = Self {
                    value: internal,
                    precision: self.precision,
                };
                result.debug_assert_valid();
                result
            }
        } else {
            let result = Self {
                value: InternalMpInt {
                    abs: self.value.abs.set_bit_to(bit, value),
                    is_positive: true,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            result
        }
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

    /// Finds the index of the first set bit.
    #[must_use]
    pub fn find_first_set_bit(&self) -> Option<usize> {
        self.find_next_set_bit(0)
    }

    /// Finds the index of the next set bit after `from`.
    #[must_use]
    pub fn find_next_set_bit(&self, from: usize) -> Option<usize> {
        if let Some(width) = self.precision.significant_bits() {
            let bits = self.to_tc_bits(width)?;
            return bits
                .find_next_set_bit(from)
                .and_then(|bit| (bit < width).then_some(bit));
        }
        if !self.is_negative() {
            return self.value.abs.find_next_set_bit(from);
        }
        let significant_bits = self.value.abs.significant_bits();
        for bit in from..significant_bits {
            if self.get_bit(bit) {
                return Some(bit);
            }
        }
        Some(from.max(significant_bits))
    }

    /// Finds the index of the first zero bit.
    #[must_use]
    pub fn find_first_zero_bit(&self) -> Option<usize> {
        if let Some(width) = self.precision.significant_bits() {
            let bits = self.to_tc_bits(width)?;
            let zero_bit = bits.find_first_zero_bit();
            return (zero_bit < width).then_some(zero_bit);
        }
        if !self.is_negative() {
            return Some(self.value.abs.find_first_zero_bit());
        }
        let trailing_zeros = self.value.abs.trailing_zeros();
        if trailing_zeros > 0 {
            return Some(0);
        }
        let start = trailing_zeros.checked_add(1)?;
        self.value.abs.find_next_set_bit(start)
    }

    /// Finds the index of the next zero bit after `from`.
    #[must_use]
    pub fn find_next_zero_bit(&self, from: usize) -> usize {
        if let Some(width) = self.precision.significant_bits() {
            let Some(bits) = self.to_tc_bits(width) else {
                return width;
            };
            let zero_bit = bits.find_next_zero_bit(from);
            return zero_bit.min(width);
        }
        if !self.is_negative() {
            return self.value.abs.find_next_zero_bit(from);
        }
        let trailing_zeros = self.value.abs.trailing_zeros();
        if from < trailing_zeros {
            return from;
        }
        let start = trailing_zeros.saturating_add(1).max(from);
        self.value
            .abs
            .find_next_set_bit(start)
            .unwrap_or(usize::MAX)
    }

    /// Extracts a bit range as a new value.
    #[must_use]
    pub fn bit_range(&self, from: usize, to: usize) -> Self {
        if self.is_negative() {
            let width = self.precision.significant_bits().unwrap_or(to);
            let bits = self
                .to_tc_bits(width.max(to))
                .unwrap_or_else(InternalMpUint::zero);
            let result = Self {
                value: InternalMpInt {
                    abs: bits.bit_range(from, to),
                    is_positive: true,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            result
        } else {
            let result = Self {
                value: InternalMpInt {
                    abs: self.value.abs.bit_range(from, to),
                    is_positive: true,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            result
        }
    }

    /// Returns the number of trailing zeros in the binary representation.
    #[must_use]
    pub fn trailing_zeros(&self) -> usize {
        self.value.abs.trailing_zeros()
    }

    /// Returns the number of trailing ones in the binary representation.
    /// Returns `None` if precision is unlimited and the integer is negative (infinite ones).
    #[must_use]
    pub fn trailing_ones(&self) -> Option<usize> {
        if let Some(width) = self.precision.significant_bits() {
            let bits = self.to_tc_bits(width)?;
            Some(bits.trailing_ones())
        } else if self.is_negative() {
            None
        } else {
            Some(self.value.abs.trailing_ones())
        }
    }

    /// Returns the number of ones in the binary representation.
    /// Returns `None` if precision is unlimited and the integer is negative (infinite ones).
    #[must_use]
    pub fn count_ones(&self) -> Option<usize> {
        if let Some(width) = self.precision.significant_bits() {
            let bits = self.to_tc_bits(width)?;
            Some(bits.count_ones())
        } else if self.is_negative() {
            None
        } else {
            Some(self.value.abs.count_ones())
        }
    }

    /// Swaps all bytes in the two's complement representation.
    /// Returns `None` if precision is unlimited (requires a bounded width).
    #[must_use]
    pub fn swap_bytes(&self) -> Option<Self> {
        let width = self.precision.significant_bits()?;
        let bits = self.to_tc_bits(width)?;
        let swapped = bits.swap_bytes(Some(width));
        Self::from_tc_bits_with_width(swapped, width)
    }
}
