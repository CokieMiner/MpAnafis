//! Core iterator aggregation implementations for the public integer types.

use core::{
    iter::{Product, Sum},
    ops::{Add, Mul},
};

use super::{ArbiInt, ArbiUint};

impl Sum for ArbiUint {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

impl Sum for ArbiInt {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

impl Product for ArbiUint {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl Product for ArbiInt {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}
