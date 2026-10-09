//! Signed integer bitwise inspection and manipulation APIs.

use core::cmp::Ordering;

use crate::error::MpError;

use super::{BoundedPrecision, InternalMpInt, InternalMpUint, MpInt, Precision};

impl MpInt {
    /// Rotates the value left within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_left(&self, n: u32, width: usize) -> Option<Self> {
        let precision = BoundedPrecision::new(width)?;
        let bits = self.value.to_tc_bits(width);
        let rotated = bits.rotate_left(n, precision);
        Some(from_tc_bits_with_precision(rotated, precision))
    }

    /// Rotates the value right within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn rotate_right(&self, n: u32, width: usize) -> Option<Self> {
        let precision = BoundedPrecision::new(width)?;
        let bits = self.value.to_tc_bits(width);
        let rotated = bits.rotate_right(n, precision);
        Some(from_tc_bits_with_precision(rotated, precision))
    }

    /// Reverses the bits within the provided width.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn reverse_bits(&self, width: usize) -> Option<Self> {
        let precision = BoundedPrecision::new(width)?;
        let bits = self.value.to_tc_bits(width);
        let reversed = bits.reverse_bits(width);
        Some(from_tc_bits_with_precision(reversed, precision))
    }

    /// Computes the bitwise NOT within the given width.
    ///
    /// Complements the width-bit residue and decodes it as a signed value.
    ///
    /// Returns `None` when `width` cannot be represented as bounded precision.
    #[must_use]
    pub fn not_with_width(&self, width: usize) -> Option<Self> {
        let precision = BoundedPrecision::new(width)?;
        let bits = self.value.to_tc_bits(width);
        let complement = bits.not(width);
        Some(from_tc_bits_with_precision(complement, precision))
    }

    /// Computes the bitwise NOT within this value's own bounded precision.
    ///
    /// # Errors
    /// Returns `MpError::WidthRequired` when the precision is unlimited.
    pub fn try_not(&self) -> Result<Self, MpError> {
        let Precision::Bounded(precision) = self.precision else {
            return Err(MpError::WidthRequired);
        };
        let width = precision.get();
        let complement = self.value.to_tc_bits(width).not(width);
        Ok(from_tc_bits_with_precision(complement, precision))
    }

    /// Returns the number of leading zeros within the current bounded width.
    ///
    /// Returns `None` if precision is unlimited (requires a bounded width).
    #[must_use]
    pub fn leading_zeros(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        Some(if self.is_negative() {
            0
        } else {
            self.value.abs.leading_zeros_for_width(width)
        })
    }

    /// Returns the number of leading ones within the current bounded width.
    ///
    /// Returns `None` if precision is unlimited (requires a bounded width).
    #[must_use]
    pub fn leading_ones(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        let bits = self.value.to_tc_bits(width);
        Some(bits.leading_ones_for_width(width))
    }

    /// Returns the number of zero bits within the current bounded width.
    ///
    /// Returns `None` if precision is unlimited (requires a bounded width).
    #[must_use]
    pub fn count_zeros(&self) -> Option<usize> {
        let width = self.precision.significant_bits()?;
        let bits = self.value.to_tc_bits(width);
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
        if let Precision::Bounded(precision) = self.precision {
            let width = precision.get();
            if bit >= width {
                return self.clone();
            }
            // bit < width preserves the encoder's width-bounded residue.
            let bits = self.value.to_tc_bits(width).set_bit_to(bit, value);
            return from_tc_bits_with_precision(bits, precision);
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

    /// Finds the index of the first set bit at or after `from`.
    #[must_use]
    pub fn find_next_set_bit(&self, from: usize) -> Option<usize> {
        let width = self.precision.significant_bits();
        if width.is_some_and(|limit| from >= limit) {
            return None;
        }
        let bit = if self.is_negative() {
            // For -a, bits below t = v2(a) are zero, bit t is one,
            // and all higher bits complement a, including sign extension.
            let trailing = self.value.abs.trailing_zeros();
            if from <= trailing {
                trailing
            } else {
                self.value.abs.find_next_zero_bit(from)
            }
        } else {
            self.value.abs.find_next_set_bit(from)?
        };
        width.is_none_or(|limit| bit < limit).then_some(bit)
    }

    /// Finds the index of the first zero bit.
    #[must_use]
    pub fn find_first_zero_bit(&self) -> Option<usize> {
        let bit = self.find_next_zero_bit(0);
        let limit = self.precision.significant_bits().unwrap_or(usize::MAX);
        (bit < limit).then_some(bit)
    }

    /// Finds the index of the first zero bit at or after `from`.
    #[must_use]
    pub fn find_next_zero_bit(&self, from: usize) -> usize {
        let limit = self.precision.significant_bits().unwrap_or(usize::MAX);
        if from >= limit {
            return limit;
        }
        if !self.is_negative() {
            return self.value.abs.find_next_zero_bit(from).min(limit);
        }
        let trailing_zeros = self.value.abs.trailing_zeros();
        if from < trailing_zeros {
            return from;
        }
        let start = trailing_zeros.saturating_add(1).max(from);
        self.value
            .abs
            .find_next_set_bit(start)
            .unwrap_or(limit)
            .min(limit)
    }

    /// Extracts a bit range as a non-negative value, widening bounded precision
    /// when the extracted bits require an additional sign bit.
    #[must_use]
    pub fn bit_range(&self, from: usize, to: usize) -> Self {
        if self.is_negative() {
            let width = self.precision.significant_bits().unwrap_or(to).max(to);
            let bits = BoundedPrecision::new(width)
                .map_or_else(InternalMpUint::zero, |precision| {
                    self.value.to_tc_bits(precision.get())
                });
            let abs = bits.bit_range(from, to);
            let precision = if let Some(limit) = self.precision.significant_bits()
                && abs.significant_bits() >= limit
            {
                abs.significant_bits()
                    .checked_add(1)
                    .and_then(Precision::new_bounded)
                    .unwrap_or(Precision::Unlimited)
            } else {
                self.precision
            };
            let result = Self {
                value: InternalMpInt {
                    abs,
                    is_positive: true,
                },
                precision,
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
        match self.precision {
            Precision::Bounded(precision) => {
                Some(self.value.to_tc_bits(precision.get()).trailing_ones())
            }
            Precision::Unlimited if self.is_negative() => None,
            Precision::Unlimited => Some(self.value.abs.trailing_ones()),
        }
    }

    /// Returns the number of ones in the binary representation.
    /// Returns `None` if precision is unlimited and the integer is negative (infinite ones).
    #[must_use]
    pub fn count_ones(&self) -> Option<usize> {
        match self.precision {
            Precision::Bounded(precision) => {
                Some(self.value.to_tc_bits(precision.get()).count_ones())
            }
            Precision::Unlimited if self.is_negative() => None,
            Precision::Unlimited => Some(self.value.abs.count_ones()),
        }
    }

    /// Swaps all bytes in the two's complement representation.
    /// Returns `None` if precision is unlimited (requires a bounded width).
    #[must_use]
    pub fn swap_bytes(&self) -> Option<Self> {
        let Precision::Bounded(precision) = self.precision else {
            return None;
        };
        let width = precision.get();
        let bits = self.value.to_tc_bits(width);
        let swapped = bits.swap_bytes(Some(width));
        // Byte reversal can set bits above a non-byte-aligned width. The
        // general decoder must truncate those bits before interpreting the sign.
        Some(from_tc_bits_with_precision(swapped, precision))
    }
}

/// Wraps the two's-complement decoder with already validated API metadata.
fn from_tc_bits_with_precision(bits: InternalMpUint, precision: BoundedPrecision) -> MpInt {
    let internal = InternalMpInt::from_tc_bits(bits, precision.get());
    let result = MpInt {
        value: internal,
        precision: Precision::Bounded(precision),
    };
    result.debug_assert_valid();
    result
}
