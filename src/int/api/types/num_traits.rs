//! `num-traits` implementations for public integer types.

use core::ops::Sub;

use num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Unsigned, Zero};

use crate::error::{ParseMpIntError, ParseMpUintError};

use super::{MpInt, MpUint};

impl Zero for MpUint {
    #[inline]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl Zero for MpInt {
    #[inline]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl One for MpUint {
    #[inline]
    fn one() -> Self {
        Self::one()
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl One for MpInt {
    #[inline]
    fn one() -> Self {
        Self::one()
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl Num for MpUint {
    type FromStrRadixErr = ParseMpUintError;
    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Self::from_str_radix(str, radix)
    }
}

impl Num for MpInt {
    type FromStrRadixErr = ParseMpIntError;
    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Self::from_str_radix(str, radix)
    }
}

impl Unsigned for MpUint {}

impl Signed for MpInt {
    #[inline]
    fn abs(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Self::zero()
        } else {
            Sub::sub(self, other)
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        self.signum()
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.is_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.is_negative()
    }
}

impl ToPrimitive for MpUint {
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.to_u64()
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.to_u128()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.to_usize()
    }
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_u64().and_then(|v| i64::try_from(v).ok())
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.to_u128().and_then(|v| i128::try_from(v).ok())
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.to_usize().and_then(|v| isize::try_from(v).ok())
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.to_f64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f32()
    }
}

impl ToPrimitive for MpInt {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_i64()
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.to_i128()
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.to_isize()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.to_u64()
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.to_u128()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.to_usize()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.to_f64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f32()
    }
}

impl FromPrimitive for MpUint {
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Self::try_from(n).ok()
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        Self::try_from(n).ok()
    }
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        Self::try_from(n).ok()
    }
}

impl FromPrimitive for MpInt {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        Some(Self::from(n))
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        Some(Self::from(n))
    }
}
