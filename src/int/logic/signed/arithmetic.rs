//! Core signed arithmetic implemented on `InternalMpInt`.

#![allow(
    clippy::same_name_method,
    reason = "InternalMpInt inherent arithmetic deliberately mirrors the corresponding operator traits."
)]

use core::cmp::max;

use super::{InternalMpInt, InternalMpUint, negate_normalized_inplace};

impl InternalMpInt {
    /// Adds two signed values.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn add(&self, other: &Self) -> Self {
        if self.is_positive == other.is_positive {
            Self {
                abs: self.abs.add(&other.abs),
                is_positive: self.is_positive,
            }
            .normalized()
        } else {
            let subtraction_width = max(self.abs.limbs().len(), other.abs.limbs().len());
            let mut difference = InternalMpUint::with_capacity(subtraction_width);
            let underflow = difference.assign_difference(&self.abs, &other.abs);
            if difference.is_zero() {
                Self::zero()
            } else if underflow {
                negate_normalized_inplace(&mut difference, subtraction_width);
                Self {
                    abs: difference,
                    is_positive: other.is_positive,
                }
            } else {
                Self {
                    abs: difference,
                    is_positive: self.is_positive,
                }
            }
        }
    }

    /// Subtracts `other` from this signed value.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn sub(&self, other: &Self) -> Self {
        if self.is_positive == other.is_positive {
            let subtraction_width = max(self.abs.limbs().len(), other.abs.limbs().len());
            let mut difference = InternalMpUint::with_capacity(subtraction_width);
            let underflow = difference.assign_difference(&self.abs, &other.abs);
            if difference.is_zero() {
                Self::zero()
            } else if underflow {
                negate_normalized_inplace(&mut difference, subtraction_width);
                Self {
                    abs: difference,
                    is_positive: !self.is_positive,
                }
            } else {
                Self {
                    abs: difference,
                    is_positive: self.is_positive,
                }
            }
        } else {
            Self {
                abs: self.abs.add(&other.abs),
                is_positive: self.is_positive,
            }
            .normalized()
        }
    }

    /// Direct shift-multiplication by `2^n`: computes `self * 2^n`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn mul_2exp(&self, shift: usize) -> Self {
        if self.abs.is_zero() {
            Self::zero()
        } else {
            Self {
                abs: self.abs.shl(shift),
                is_positive: self.is_positive,
            }
            .normalized()
        }
    }

    /// Multiplies two signed values.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            abs: self.abs.mul(&other.abs),
            is_positive: self.is_positive == other.is_positive,
        }
        .normalized()
    }

    /// Squares this signed value.
    #[inline]
    #[must_use]
    pub fn square(&self) -> Self {
        Self {
            abs: self.abs.square(),
            is_positive: true,
        }
        .normalized()
    }

    /// Multiplies this value in place by `other`.
    #[inline]
    #[track_caller]
    pub fn mul_assign(&mut self, other: &Self) {
        let product_is_positive = self.is_positive == other.is_positive;
        self.abs.mul_assign(&other.abs);
        self.is_positive = self.abs.is_zero() || product_is_positive;
    }

    /// Multiplies two owned values while reusing their storage.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn mul_into(self, other: Self) -> Self {
        let product_is_positive = self.is_positive == other.is_positive;
        Self {
            abs: self.abs.mul_into(other.abs),
            is_positive: product_is_positive,
        }
        .normalized()
    }
}
