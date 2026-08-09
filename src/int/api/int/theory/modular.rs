//! Modular arithmetic and reduction APIs for signed integers.

use super::{InternalMpInt, MpInt};

impl MpInt {
    /// Returns `(self + other) % modulus` on the absolute values.
    #[must_use]
    pub fn add_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.add_mod(&other.value.abs, &modulus.value.abs),
                is_positive: true,
            },
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self - other) % modulus` on the absolute values.
    #[must_use]
    pub fn sub_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.sub_mod(&other.value.abs, &modulus.value.abs),
                is_positive: true,
            },
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self * other) % modulus` on the absolute values.
    #[must_use]
    pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.mul_mod(&other.value.abs, &modulus.value.abs),
                is_positive: true,
            },
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self ^ exp) % modulus` on the absolute values.
    /// If `exp` is negative, `invert(modulus)?` is used as the base.
    #[must_use]
    pub fn pow_mod(&self, exp: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        let base = if exp.is_negative() {
            self.invert(modulus)?
        } else {
            self.clone()
        };
        let p = self
            .precision
            .combine_for_binary_op(exp.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: InternalMpInt {
                abs: base.value.abs.pow_mod(&exp.value.abs, &modulus.value.abs),
                is_positive: true,
            },
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the modular inverse of the absolute value.
    #[must_use]
    pub fn invert(&self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        self.value.abs.invert(&modulus.value.abs).map(|v| {
            let p = self.precision.combine_for_binary_op(modulus.precision);
            let result = Self {
                value: InternalMpInt {
                    abs: v,
                    is_positive: true,
                },
                precision: p,
            };
            result.debug_assert_valid();
            result
        })
    }

    /// Computes the square `self * self`.
    ///
    /// # Panics
    ///
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn square(&self) -> Self {
        let result = Self {
            value: self.value.square(),
            precision: self.precision,
        };
        result.assert_fits("square");
        result.debug_assert_valid();
        result
    }

    /// Computes Montgomery multiplication `(|self| * |other| * R^{-1}) mod |modulus|`.
    ///
    /// Returns `None` if `modulus` is even or zero. Result is always non-negative.
    #[must_use]
    pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if !modulus.value.abs.is_odd() {
            return None;
        }
        let result = Self {
            value: InternalMpInt {
                abs: self
                    .value
                    .abs
                    .montgomery_mul(&other.value.abs, &modulus.value.abs),
                is_positive: true,
            },
            precision: self
                .precision
                .combine_for_binary_op(other.precision)
                .combine_for_binary_op(modulus.precision),
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Performs modular reduction `|self| mod |modulus|` using Barrett reduction.
    ///
    /// Returns `None` if `modulus` is zero. Result is always non-negative.
    #[must_use]
    pub fn barrett_reduce(&self, modulus: &Self) -> Option<Self> {
        if modulus.value.abs.is_zero() {
            return None;
        }
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.barrett_reduce(&modulus.value.abs),
                is_positive: true,
            },
            precision: self.precision.combine_for_binary_op(modulus.precision),
        };
        result.debug_assert_valid();
        Some(result)
    }
}
