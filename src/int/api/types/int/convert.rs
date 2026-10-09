//! Signed integer inherent conversion APIs.

#![cfg_attr(
    feature = "num-traits",
    expect(
        clippy::same_name_method,
        reason = "Inherent conversions and radix parsing intentionally mirror num_traits conveniences"
    )
)]

use alloc::{string::String, vec::Vec};

use crate::error::{ParseMpIntError, ParseMpIntErrorKind, ParseMpUintErrorKind};

use super::{InternalMpInt, InternalMpUint, MpInt, Precision};

impl MpInt {
    /// Converts the value to a `u64`, or `None` if it does not fit or is negative.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_u64()
        }
    }

    /// Converts the value to a `u128`, or `None` if it does not fit or is negative.
    #[must_use]
    pub fn to_u128(&self) -> Option<u128> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_u128()
        }
    }

    /// Converts the value to a `usize`, or `None` if it does not fit or is negative.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_usize()
        }
    }

    /// Converts the value to an `i64`, or `None` if it does not fit.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        let abs = self.value.abs.to_u64()?;
        if !self.is_negative() {
            i64::try_from(abs).ok()
        } else if abs == (1_u64 << 63) {
            Some(i64::MIN)
        } else {
            i64::try_from(abs).ok().map(i64::wrapping_neg)
        }
    }

    /// Converts the value to an `i128`, or `None` if it does not fit.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        let abs = self.value.abs.to_u128()?;
        if !self.is_negative() {
            i128::try_from(abs).ok()
        } else if abs == (1_u128 << 127) {
            Some(i128::MIN)
        } else {
            i128::try_from(abs).ok().map(i128::wrapping_neg)
        }
    }

    /// Converts the value to an `isize`, or `None` if it does not fit.
    #[must_use]
    pub fn to_isize(&self) -> Option<isize> {
        let abs = self.value.abs.to_usize()?;
        if !self.is_negative() {
            isize::try_from(abs).ok()
        } else if abs == (1_usize << (usize::BITS - 1)) {
            Some(isize::MIN)
        } else {
            isize::try_from(abs).ok().map(isize::wrapping_neg)
        }
    }

    /// Converts the value to an `f64`.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        if self.is_negative() {
            self.value.abs.to_f64().map(|v| -v)
        } else {
            self.value.abs.to_f64()
        }
    }

    /// Converts the value to an `f32`.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        if self.is_negative() {
            self.value.abs.to_f32().map(|v| -v)
        } else {
            self.value.abs.to_f32()
        }
    }

    /// Parses an `MpInt` from a string slice in the given radix.
    ///
    /// # Errors
    /// Returns a `ParseMpIntError` if the string contains invalid digits,
    /// an invalid radix is provided, or the value is empty or too large.
    pub fn from_str_radix(str: &str, radix: u32) -> Result<Self, ParseMpIntError> {
        let (is_positive, rest) = str.strip_prefix('-').map_or_else(
            || {
                str.strip_prefix('+')
                    .map_or((true, str), |stripped| (true, stripped))
            },
            |stripped| (false, stripped),
        );
        let abs = InternalMpUint::from_str_radix(rest, radix).map_err(|e| {
            let kind = match e.kind() {
                ParseMpUintErrorKind::Empty => ParseMpIntErrorKind::Empty,
                ParseMpUintErrorKind::InvalidDigit | ParseMpUintErrorKind::Negative => {
                    ParseMpIntErrorKind::InvalidDigit
                }
                ParseMpUintErrorKind::InvalidRadix => ParseMpIntErrorKind::InvalidRadix,
                ParseMpUintErrorKind::TooLarge => ParseMpIntErrorKind::TooLarge,
            };
            ParseMpIntError::new(kind)
        })?;
        let final_is_positive = is_positive || abs.is_zero();
        let internal = InternalMpInt {
            abs,
            is_positive: final_is_positive,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let Some(precision) = Precision::checked_for_ambient_construction(required) else {
            return Err(ParseMpIntError::new(ParseMpIntErrorKind::TooLarge));
        };
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }

    /// Formats the `MpInt` into a string in radix `2..=36`.
    ///
    /// # Panics
    ///
    /// Panics if `radix` is outside `2..=36`.
    #[must_use]
    #[track_caller]
    pub fn to_string_radix(&self, radix: u32) -> String {
        let mut s = self.value.abs.to_string_radix(radix);
        if self.is_negative() {
            s.insert(0, '-');
        }
        s
    }

    /// Returns the integer as a two's complement big-endian byte vector
    /// (most significant byte first), with the minimum number of bytes needed
    /// to preserve sign.
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let mut bytes = self.value.abs.to_be_bytes();
        let negative = self.is_negative();
        if negative {
            // In -a, zero low bytes remain zero, the first nonzero byte
            // is negated modulo 256, and every higher byte is complemented.
            let zero_bytes = self.value.abs.trailing_zeros() >> 3;
            let mut remaining = bytes.iter_mut().rev().skip(zero_bytes);
            if let Some(last) = remaining.next() {
                *last = last.wrapping_neg();
                for byte in remaining {
                    *byte = !*byte;
                }
            }
        }
        if bytes
            .first()
            .is_some_and(|byte| (byte & 0x80 != 0) != negative)
        {
            bytes.insert(0, if negative { 0xFF } else { 0 });
        }
        bytes
    }

    /// Returns the integer as a two's complement little-endian byte vector
    /// (least significant byte first), with the minimum number of bytes needed
    /// to preserve sign.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = self.value.abs.to_le_bytes();
        let negative = self.is_negative();
        if negative {
            // The carry of !a + 1 crosses only the low zero bytes. Once
            // consumed, all remaining bytes are independent complements.
            let zero_bytes = self.value.abs.trailing_zeros() >> 3;
            let mut remaining = bytes.iter_mut().skip(zero_bytes);
            if let Some(first) = remaining.next() {
                *first = first.wrapping_neg();
                for byte in remaining {
                    *byte = !*byte;
                }
            }
        }
        if bytes
            .last()
            .is_some_and(|byte| (byte & 0x80 != 0) != negative)
        {
            bytes.push(if negative { 0xFF } else { 0 });
        }
        bytes
    }

    /// Constructs an `MpInt` from a two's complement little-endian byte slice.
    ///
    /// Redundant sign-extension bytes are ignored.
    ///
    /// # Panics
    ///
    /// Panics when the canonical input width cannot be represented by
    /// `usize` on the target platform.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let canonical_bytes = trim_signed_le_bytes(bytes);
        if canonical_bytes.is_empty() {
            return Self::zero();
        }
        // An addressable byte slice can have a bit width greater than usize::MAX.
        let width = canonical_bytes
            .len()
            .checked_mul(8)
            .expect("canonical byte input width must fit in usize");
        let tc_uint = InternalMpUint::from_le_bytes(canonical_bytes);
        let internal = InternalMpInt::from_tc_bits(tc_uint, width);
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Constructs an `MpInt` from a two's complement big-endian byte slice.
    ///
    /// Redundant sign-extension bytes are ignored.
    ///
    /// # Panics
    ///
    /// Panics when the canonical input width cannot be represented by
    /// `usize` on the target platform.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let canonical_bytes = trim_signed_be_bytes(bytes);
        if canonical_bytes.is_empty() {
            return Self::zero();
        }
        // An addressable byte slice can have a bit width greater than usize::MAX.
        let width = canonical_bytes
            .len()
            .checked_mul(8)
            .expect("canonical byte input width must fit in usize");
        let tc_uint = InternalMpUint::from_be_bytes(canonical_bytes);
        let internal = InternalMpInt::from_tc_bits(tc_uint, width);
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}

fn trim_signed_le_bytes(mut bytes: &[u8]) -> &[u8] {
    while let [prefix @ .., extension] = bytes {
        let Some(next) = prefix.last() else {
            break;
        };
        let redundant_zero = *extension == 0 && next & 0x80 == 0;
        let redundant_ones = *extension == 0xFF && next & 0x80 != 0;
        if !redundant_zero && !redundant_ones {
            break;
        }
        bytes = prefix;
    }
    bytes
}

fn trim_signed_be_bytes(mut bytes: &[u8]) -> &[u8] {
    while let [extension, suffix @ ..] = bytes {
        let Some(next) = suffix.first() else {
            break;
        };
        let redundant_zero = *extension == 0 && next & 0x80 == 0;
        let redundant_ones = *extension == 0xFF && next & 0x80 != 0;
        if !redundant_zero && !redundant_ones {
            break;
        }
        bytes = suffix;
    }
    bytes
}
