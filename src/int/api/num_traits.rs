//! `num-traits` implementations for public integer types.

#![cfg(feature = "num-traits")]

use ::num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Unsigned, Zero};

use crate::error::{ParseMpIntError, ParseMpUintError};

use super::{MpInt, MpUint};

impl Zero for MpUint {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl Zero for MpInt {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        self.value.abs.is_zero()
    }
}

impl One for MpUint {
    fn one() -> Self {
        Self::one()
    }
    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

impl One for MpInt {
    fn one() -> Self {
        Self::one()
    }
    fn is_one(&self) -> bool {
        self.value.abs.is_one() && self.value.is_positive
    }
}

impl Num for MpUint {
    type FromStrRadixErr = ParseMpUintError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Self::from_str_radix(str, radix)
    }
}

impl Num for MpInt {
    type FromStrRadixErr = ParseMpIntError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Self::from_str_radix(str, radix)
    }
}

impl Unsigned for MpUint {}

impl Signed for MpInt {
    fn abs(&self) -> Self {
        self.abs()
    }
    fn abs_sub(&self, other: &Self) -> Self {
        Self::abs_sub(self, other)
    }
    fn signum(&self) -> Self {
        self.signum()
    }
    fn is_positive(&self) -> bool {
        self.is_positive()
    }
    fn is_negative(&self) -> bool {
        self.is_negative()
    }
}

impl ToPrimitive for MpUint {
    fn to_u64(&self) -> Option<u64> {
        self.to_u64()
    }
    fn to_u128(&self) -> Option<u128> {
        self.to_u128()
    }
    fn to_usize(&self) -> Option<usize> {
        self.to_usize()
    }
    fn to_i64(&self) -> Option<i64> {
        self.to_u64().and_then(|v| i64::try_from(v).ok())
    }
    fn to_i128(&self) -> Option<i128> {
        self.to_u128().and_then(|v| i128::try_from(v).ok())
    }
    fn to_isize(&self) -> Option<isize> {
        self.to_usize().and_then(|v| isize::try_from(v).ok())
    }
    fn to_f64(&self) -> Option<f64> {
        self.to_f64()
    }
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
