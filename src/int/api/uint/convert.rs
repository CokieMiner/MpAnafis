//! Unsigned integer inherent conversion APIs.

use alloc::{string::String, vec::Vec};

use crate::error::ParseMpUintError;

use super::{InternalMpUint, MpUint, Precision};

impl MpUint {
    // Convert
    /// Converts the value to a `u64`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }

    /// Converts the value to a `u128`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_u128(&self) -> Option<u128> {
        self.value.to_u128()
    }

    /// Converts the value to a `usize`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_usize(&self) -> Option<usize> {
        self.value.to_usize()
    }

    /// Converts the value to an `i64`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn to_i64(&self) -> Option<i64> {
        self.to_u64().and_then(|v| i64::try_from(v).ok())
    }

    /// Converts the value to an `i128`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn to_i128(&self) -> Option<i128> {
        self.to_u128().and_then(|v| i128::try_from(v).ok())
    }

    /// Converts the value to an `isize`, or `None` if it does not fit.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn to_isize(&self) -> Option<isize> {
        self.to_usize().and_then(|v| isize::try_from(v).ok())
    }

    /// Parses an `MpUint` from a string slice in the given radix.
    ///
    /// # Errors
    ///
    /// Returns `ParseMpUintError` if the string contains invalid digits for
    /// the given radix, the radix is out of range, or the value is too large.
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseMpUintError> {
        let internal = InternalMpUint::from_str_radix(s, radix)?;
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }

    /// Formats the `MpUint` into a string in radix `2..=36`.
    ///
    /// # Panics
    ///
    /// Panics if `radix` is outside `2..=36`.
    #[must_use]
    #[track_caller]
    pub fn to_string_radix(&self, radix: u32) -> String {
        self.value.to_string_radix(radix)
    }

    // Float conversion
    // ------------------------------------------------------------------

    /// Converts the value to `f64`. Returns `None` if the value is too large
    /// to be represented in `f64`.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }

    /// Converts the value to `f32`. Returns `None` if the value is too large
    /// to be represented in `f32`.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn to_f32(&self) -> Option<f32> {
        self.value.to_f32()
    }

    // Byte conversions
    // ------------------------------------------------------------------

    /// Returns the integer as a little-endian byte vector (least significant byte first).
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.value.to_le_bytes()
    }

    /// Returns the integer as a big-endian byte vector (most significant byte first).
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        self.value.to_be_bytes()
    }

    /// Constructs an `MpUint` from a little-endian byte slice.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let internal = InternalMpUint::from_le_bytes(bytes);
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Constructs an `MpUint` from a big-endian byte slice.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let internal = InternalMpUint::from_be_bytes(bytes);
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let precision = Precision::for_ambient_construction(required);
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        result
    }
}
