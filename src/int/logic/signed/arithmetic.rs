//! Core signed arithmetic implemented on `InternalMpInt`.

use core::cmp::max;

use super::{InternalMpInt, InternalMpUint};

impl InternalMpInt {
    /// Adds two signed values $a + b$.
    ///
    /// When operand signs match, computes $|a| + |b|$ under the shared sign.
    /// When signs differ, evaluates $|a| - |b|$ via `assign_difference`. Upon underflow,
    /// inverts the residue via two's complement to obtain $|b| - |a|$ with the sign of $b$.
    /// Cancellation returns positive zero.
    #[inline]
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        if self.is_positive == other.is_positive {
            Self {
                abs: self.abs.add(&other.abs),
                is_positive: self.is_positive,
            }
        } else {
            let subtraction_width = max(self.abs.limbs().len(), other.abs.limbs().len());
            let mut difference = InternalMpUint::with_capacity(subtraction_width);
            let underflow = difference.assign_difference(&self.abs, &other.abs);
            if difference.is_zero() {
                Self::zero()
            } else if underflow {
                Self::negate_normalized_inplace(&mut difference, subtraction_width);
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

    /// Proves that a sum fits a signed width using only magnitude widths.
    ///
    /// Equal signs require room for a carry and the sign bit. Opposite signs
    /// subtract magnitudes, so the larger magnitude needs only the sign bit.
    /// A false result requires exact validation; cancellation and the negative
    /// endpoint can still produce a representable sum.
    #[inline]
    #[must_use]
    pub fn sum_fits_by_width(&self, other: &Self, bits: usize) -> bool {
        debug_assert!(bits != 0, "bounded signed precision is nonzero");
        let guard_bits = if self.is_positive == other.is_positive {
            2
        } else {
            1
        };
        let magnitude_bits = bits.saturating_sub(guard_bits);
        self.abs.fits_in_bits(magnitude_bits) && other.abs.fits_in_bits(magnitude_bits)
    }

    /// Subtracts `other` from this signed value ($a - b$).
    ///
    /// When operand signs differ, computes $|a| + |b|$ under the sign of $a$.
    /// When signs match, evaluates $|a| - |b|$ via `assign_difference`. Upon underflow,
    /// inverts the residue to obtain $|b| - |a|$ with inverted sign $\neg \operatorname{sgn}(a)$.
    /// Equal magnitudes with equal signs return positive zero.
    #[inline]
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        if self.is_positive == other.is_positive {
            let subtraction_width = max(self.abs.limbs().len(), other.abs.limbs().len());
            let mut difference = InternalMpUint::with_capacity(subtraction_width);
            let underflow = difference.assign_difference(&self.abs, &other.abs);
            if difference.is_zero() {
                Self::zero()
            } else if underflow {
                Self::negate_normalized_inplace(&mut difference, subtraction_width);
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
        }
    }

    /// Multiplies by `2^shift`, preserving the sign.
    #[inline]
    #[must_use]
    pub fn mul_2exp(&self, shift: usize) -> Self {
        Self {
            abs: self.abs.shl(shift),
            is_positive: self.is_positive,
        }
    }

    /// Multiplies two signed values ($a \cdot b$).
    ///
    /// Multiplies magnitudes. A nonzero product is positive exactly when the
    /// operand signs agree; a zero product is positive.
    #[inline]
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let abs = self.abs.mul(&other.abs);
        let product_is_positive = self.is_positive == other.is_positive;
        Self {
            is_positive: abs.is_zero() || product_is_positive,
            abs,
        }
    }

    /// Squares this signed value ($a^2$).
    ///
    /// For all $a \in \mathbb{Z}$, $a^2 \ge 0$. Sign is strictly positive.
    #[inline]
    #[must_use]
    pub fn square(&self) -> Self {
        Self {
            abs: self.abs.square(),
            is_positive: true,
        }
    }

    /// Multiplies this value in place by `other` ($a \leftarrow a \cdot b$).
    #[inline]
    pub fn mul_assign(&mut self, other: &Self) {
        let product_is_positive = self.is_positive == other.is_positive;
        self.abs.mul_assign(&other.abs);
        self.is_positive = self.abs.is_zero() || product_is_positive;
    }

    /// Multiplies two owned values while reusing allocated storage.
    #[inline]
    #[must_use]
    pub fn mul_into(self, other: Self) -> Self {
        let product_is_positive = self.is_positive == other.is_positive;
        let abs = self.abs.mul_into(other.abs);
        Self {
            is_positive: abs.is_zero() || product_is_positive,
            abs,
        }
    }
}
