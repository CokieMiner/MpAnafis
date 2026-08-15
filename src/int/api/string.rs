//! String formatting and parsing trait implementations.

use core::{
    fmt::{Binary, Debug, Display, Formatter, LowerHex, Octal, Result as FmtResult, UpperHex},
    str::FromStr,
};

use crate::error::{ParseMpIntError, ParseMpIntErrorKind, ParseMpUintError, ParseMpUintErrorKind};

use super::{
    AmbientPrecision, DebugVerbose, InternalMpInt, InternalMpUint, MpInt, MpUint, Precision,
    PrecisionContext,
};

impl Debug for DebugVerbose<'_, MpInt> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "MpInt({}, precision: {:?})", self.0, self.0.precision)
    }
}

impl Debug for MpUint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{self}")
    }
}

impl Debug for MpInt {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{self}")
    }
}

macro_rules! impl_fmt_uint {
    ($t:ty) => {
        impl Display for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                Display::fmt(&self.value, f)
            }
        }
        impl LowerHex for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.to_string_radix(16);
                f.pad_integral(true, "0x", &s)
            }
        }
        impl UpperHex for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.to_string_radix(16).to_ascii_uppercase();
                f.pad_integral(true, "0x", &s)
            }
        }
        impl Octal for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.to_string_radix(8);
                f.pad_integral(true, "0o", &s)
            }
        }
        impl Binary for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.to_string_radix(2);
                f.pad_integral(true, "0b", &s)
            }
        }
    };
}

macro_rules! impl_fmt_int {
    ($t:ty) => {
        impl Display for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                if !self.value.is_positive {
                    write!(f, "-")?;
                }
                Display::fmt(&self.value.abs, f)
            }
        }
        impl LowerHex for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.abs.to_string_radix(16);
                f.pad_integral(self.value.is_positive, "0x", &s)
            }
        }
        impl UpperHex for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.abs.to_string_radix(16).to_ascii_uppercase();
                f.pad_integral(self.value.is_positive, "0x", &s)
            }
        }
        impl Octal for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.abs.to_string_radix(8);
                f.pad_integral(self.value.is_positive, "0o", &s)
            }
        }
        impl Binary for $t {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                let s = self.value.abs.to_string_radix(2);
                f.pad_integral(self.value.is_positive, "0b", &s)
            }
        }
    };
}

impl FromStr for MpUint {
    type Err = ParseMpUintError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let internal = InternalMpUint::from_str_radix(s, 10)?;
        let required = internal.required_unsigned_bits_for_bounded_storage();
        let ambient = PrecisionContext::active();
        let precision = match ambient {
            AmbientPrecision::Unlimited | AmbientPrecision::Unset => Precision::Unlimited,
            AmbientPrecision::Bounded(n) => {
                if required > n.get() {
                    return Err(ParseMpUintError {
                        kind: ParseMpUintErrorKind::TooLarge,
                    });
                }
                Precision::Bounded(n)
            }
        };
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}

impl FromStr for MpInt {
    type Err = ParseMpIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (is_positive, rest) = s.strip_prefix('-').map_or_else(
            || {
                s.strip_prefix('+')
                    .map_or((true, s), |stripped| (true, stripped))
            },
            |stripped| (false, stripped),
        );

        let abs = InternalMpUint::from_str_radix(rest, 10).map_err(|e| ParseMpIntError {
            kind: match e.kind {
                ParseMpUintErrorKind::Empty => ParseMpIntErrorKind::Empty,
                ParseMpUintErrorKind::InvalidDigit | ParseMpUintErrorKind::Negative => {
                    ParseMpIntErrorKind::InvalidDigit
                }
                ParseMpUintErrorKind::InvalidRadix => ParseMpIntErrorKind::InvalidRadix,
                ParseMpUintErrorKind::TooLarge => ParseMpIntErrorKind::TooLarge,
            },
        })?;

        let is_pos = if abs.is_zero() { true } else { is_positive };
        let internal = InternalMpInt {
            abs,
            is_positive: is_pos,
        };
        let required = internal.required_signed_bits_for_bounded_storage();
        let ambient = PrecisionContext::active();
        let precision = match ambient {
            AmbientPrecision::Unlimited | AmbientPrecision::Unset => Precision::Unlimited,
            AmbientPrecision::Bounded(n) => {
                if required > n.get() {
                    return Err(ParseMpIntError {
                        kind: ParseMpIntErrorKind::TooLarge,
                    });
                }
                Precision::Bounded(n)
            }
        };
        let result = Self {
            value: internal,
            precision,
        };
        result.debug_assert_valid();
        Ok(result)
    }
}

impl_fmt_uint!(MpUint);
impl_fmt_int!(MpInt);
