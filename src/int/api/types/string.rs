//! String formatting and parsing trait implementations.

use core::{
    fmt::{Binary, Debug, Display, Formatter, LowerHex, Octal, Result as FmtResult, UpperHex},
    str::FromStr,
};

use crate::error::{ParseMpIntError, ParseMpUintError};

use super::{DebugVerbose, MpInt, MpUint};

impl Debug for DebugVerbose<'_, MpUint> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "MpUint({}, precision: {:?})", self.0, self.0.precision)
    }
}

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
                let mut s = self.value.to_string_radix(16);
                s.make_ascii_uppercase();
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
                if f.width().is_some() || f.precision().is_some() || f.sign_plus() {
                    let digits = self.value.abs.to_string_radix(10);
                    return f.pad_integral(self.value.is_positive, "", &digits);
                }
                // Canonical zero has a positive sign, so only nonzero values emit '-'.
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
                let mut s = self.value.abs.to_string_radix(16);
                s.make_ascii_uppercase();
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
        Self::from_str_radix(s, 10)
    }
}

impl FromStr for MpInt {
    type Err = ParseMpIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(s, 10)
    }
}

impl_fmt_uint!(MpUint);
impl_fmt_int!(MpInt);
