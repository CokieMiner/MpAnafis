//! Modular arithmetic and reduction APIs for unsigned integers.

use super::MpUint;

impl MpUint {
    /// Returns `(self + other) % modulus`, or `None` if the modulus is zero.
    #[must_use]
    pub fn add_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: self.value.add_mod(&other.value, &modulus.value),
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self - other) % modulus`, or `None` if the modulus is zero.
    #[must_use]
    pub fn sub_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: self.value.sub_mod(&other.value, &modulus.value),
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self * other) % modulus`, or `None` if the modulus is zero.
    #[must_use]
    pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: self.value.mul_mod(&other.value, &modulus.value),
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `(self ^ exp) % modulus`, or `None` if the modulus is zero.
    #[must_use]
    pub fn pow_mod(&self, exp: &Self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        let p = self
            .precision
            .combine_for_binary_op(exp.precision)
            .combine_for_binary_op(modulus.precision);
        let result = Self {
            value: self.value.pow_mod(&exp.value, &modulus.value),
            precision: p,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the modular multiplicative inverse of `self` modulo `modulus`, or `None` if no inverse exists.
    #[must_use]
    pub fn invert(&self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        self.value.invert(&modulus.value).map(|v| {
            let p = self.precision.combine_for_binary_op(modulus.precision);
            let result = Self {
                value: v,
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

    /// Performs Montgomery multiplication `(self * other * R^{-1}) mod modulus`.
    ///
    /// Returns `None` if `modulus` is even or zero.
    #[must_use]
    pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Option<Self> {
        if !modulus.value.is_odd() {
            return None;
        }
        let result = Self {
            value: self.value.montgomery_mul(&other.value, &modulus.value),
            precision: self
                .precision
                .combine_for_binary_op(other.precision)
                .combine_for_binary_op(modulus.precision),
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Performs modular reduction `self % modulus` using Barrett reduction.
    ///
    /// Returns `None` if `modulus` is zero.
    #[must_use]
    pub fn barrett_reduce(&self, modulus: &Self) -> Option<Self> {
        if modulus.value.is_zero() {
            return None;
        }
        let result = Self {
            value: self.value.barrett_reduce(&modulus.value),
            precision: self.precision.combine_for_binary_op(modulus.precision),
        };
        result.debug_assert_valid();
        Some(result)
    }
}
