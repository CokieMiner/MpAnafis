//! Signed integer inherent conversion APIs.

use alloc::{string::String, vec::Vec};

use crate::error::{ParseMpIntError, ParseMpIntErrorKind, ParseMpUintErrorKind};

use super::{InternalMpInt, InternalMpUint, MpInt, MpUint, Precision};

impl MpInt {
    /// Converts the value to a `u64`, or `None` if it does not fit or is negative.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_u64(&self) -> Option<u64> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_u64()
        }
    }

    /// Converts the value to a `u128`, or `None` if it does not fit or is negative.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_u128(&self) -> Option<u128> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_u128()
        }
    }

    /// Converts the value to a `usize`, or `None` if it does not fit or is negative.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_usize(&self) -> Option<usize> {
        if self.is_negative() {
            None
        } else {
            self.value.abs.to_usize()
        }
    }

    /// Converts the value to an `i64`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_f64(&self) -> Option<f64> {
        if self.is_negative() {
            self.value.abs.to_f64().map(|v| -v)
        } else {
            self.value.abs.to_f64()
        }
    }

    /// Converts the value to an `f32`.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn from_str_radix(str: &str, radix: u32) -> Result<Self, ParseMpIntError> {
        let (is_positive, rest) = str.strip_prefix('-').map_or_else(
            || {
                str.strip_prefix('+')
                    .map_or((true, str), |stripped| (true, stripped))
            },
            |stripped| (false, stripped),
        );
        let uint = MpUint::from_str_radix(rest, radix).map_err(|e| {
            let kind = match e.kind {
                ParseMpUintErrorKind::Empty | ParseMpUintErrorKind::Negative => {
                    ParseMpIntErrorKind::Empty
                }
                ParseMpUintErrorKind::InvalidDigit => ParseMpIntErrorKind::InvalidDigit,
                ParseMpUintErrorKind::InvalidRadix => ParseMpIntErrorKind::InvalidRadix,
                ParseMpUintErrorKind::TooLarge => ParseMpIntErrorKind::TooLarge,
            };
            ParseMpIntError { kind }
        })?;
        let final_is_positive = is_positive || uint.is_zero();
        let internal = InternalMpInt {
            abs: uint.value,
            is_positive: final_is_positive,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
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

    // Byte conversions
    // ------------------------------------------------------------------

    /// Returns the integer as a two's complement little-endian byte vector
    /// (least significant byte first), with the minimum number of bytes needed
    /// to preserve sign.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        if self.is_zero() {
            return Vec::new();
        }
        let num_bytes = self
            .value
            .required_signed_bits_for_bounded_storage()
            .div_ceil(8);
        let width = num_bytes.wrapping_mul(8);
        let tc_uint = self.value.to_tc_bits(width);
        let mut bytes = tc_uint.to_le_bytes();
        bytes.resize(num_bytes, if self.is_negative() { 0xFF } else { 0x00 });
        bytes
    }

    /// Returns the integer as a two's complement big-endian byte vector
    /// (most significant byte first), with the minimum number of bytes needed
    /// to preserve sign.
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        if self.is_zero() {
            return Vec::new();
        }
        let num_bytes = self
            .value
            .required_signed_bits_for_bounded_storage()
            .div_ceil(8);
        let width = num_bytes.wrapping_mul(8);
        let tc_uint = self.value.to_tc_bits(width);
        // The magnitude encoder trims leading zero bytes, so the sign extension
        // has to be restored here. Little-endian gets this from `resize`, which
        // appends; big-endian needs the same bytes at the front instead.
        let magnitude = tc_uint.to_be_bytes();
        let sign_extension = num_bytes.saturating_sub(magnitude.len());
        let mut bytes = Vec::with_capacity(num_bytes);
        bytes.resize(sign_extension, if self.is_negative() { 0xFF } else { 0x00 });
        bytes.extend_from_slice(&magnitude);
        bytes
    }

    /// Constructs an `MpInt` from a two's complement little-endian byte slice.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }
        let width = bytes.len().wrapping_mul(8);
        let tc_uint = InternalMpUint::from_le_bytes(bytes);
        let internal = if bytes.last().is_some_and(|&b| b & 0x80 != 0) {
            let mut abs = tc_uint.not(width);
            abs.increment();
            InternalMpInt {
                abs,
                is_positive: false,
            }
        } else if tc_uint.is_zero() {
            InternalMpInt::zero()
        } else {
            InternalMpInt {
                abs: tc_uint,
                is_positive: true,
            }
        };
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
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }
        let width = bytes.len().wrapping_mul(8);
        let tc_uint = InternalMpUint::from_be_bytes(bytes);
        let internal = if bytes.first().is_some_and(|&b| b & 0x80 != 0) {
            let mut abs = tc_uint.not(width);
            abs.increment();
            InternalMpInt {
                abs,
                is_positive: false,
            }
        } else if tc_uint.is_zero() {
            InternalMpInt::zero()
        } else {
            InternalMpInt {
                abs: tc_uint,
                is_positive: true,
            }
        };
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
