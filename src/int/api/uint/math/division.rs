//! Unsigned division and divisibility APIs.

use super::MpUint;

impl MpUint {
    /// Returns the quotient and remainder of division, or `None` when `rhs` is zero.
    #[must_use]
    pub fn div_rem(&self, rhs: &Self) -> Option<(Self, Self)> {
        if rhs.value.is_zero() {
            return None;
        }
        let (quot, rem) = self.value.div_rem(&rhs.value);
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let quotient = Self {
            value: quot,
            precision: p,
        };
        quotient.debug_assert_valid();
        let remainder = Self {
            value: rem,
            precision: p,
        };
        remainder.debug_assert_valid();
        Some((quotient, remainder))
    }

    /// Returns `true` if `self` is divisible by `other` (i.e., `self % other == 0`).
    ///
    /// By convention, zero is divisible by zero; no nonzero value is divisible
    /// by zero.
    #[must_use]
    pub fn is_divisible_by(&self, other: &Self) -> bool {
        self.value.is_divisible_by(&other.value)
    }

    /// Returns `true` if `self` divides `other` (i.e., `other % self == 0`).
    ///
    /// By convention, zero divides zero but no other value.
    #[must_use]
    pub fn is_divisor_of(&self, other: &Self) -> bool {
        other.is_divisible_by(self)
    }

    /// Truncating division (identical to `/`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn div_trunc(&self, rhs: &Self) -> Self {
        // Routed through the quotient-only entry point rather than `div_rem`:
        // discarding a remainder still costs the work of producing it, which is
        // the whole width of the divisor on every quotient limb.
        self.checked_div(rhs)
            .expect("div_trunc requires a non-zero divisor")
    }

    /// Checked truncating division.
    #[must_use]
    pub fn checked_div_trunc(&self, rhs: &Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Truncating remainder (identical to `%`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn rem_trunc(&self, rhs: &Self) -> Self {
        self.checked_rem(rhs)
            .expect("rem_trunc requires a non-zero divisor")
    }

    /// Checked truncating remainder.
    #[must_use]
    pub fn checked_rem_trunc(&self, rhs: &Self) -> Option<Self> {
        self.checked_rem(rhs)
    }

    /// Returns the quotient and remainder of Euclidean division
    /// (for unsigned integers, identical to `div_rem`).
    #[must_use]
    pub fn div_rem_euclid(&self, rhs: &Self) -> Option<(Self, Self)> {
        self.div_rem(rhs)
    }

    /// Euclidean division (for unsigned integers, identical to `/`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn div_euclid(&self, rhs: &Self) -> Self {
        self.div_trunc(rhs)
    }

    /// Checked Euclidean division.
    #[must_use]
    pub fn checked_div_euclid(&self, rhs: &Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Euclidean remainder (for unsigned integers, identical to `%`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn rem_euclid(&self, rhs: &Self) -> Self {
        self.rem_trunc(rhs)
    }

    /// Checked Euclidean remainder.
    #[must_use]
    pub fn checked_rem_euclid(&self, rhs: &Self) -> Option<Self> {
        self.checked_rem(rhs)
    }

    /// Returns the quotient and remainder of floor division
    /// (for unsigned integers, identical to `div_rem`).
    #[must_use]
    pub fn div_rem_floor(&self, rhs: &Self) -> Option<(Self, Self)> {
        self.div_rem(rhs)
    }

    /// Floor division (for unsigned integers, identical to `/`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn div_floor(&self, rhs: &Self) -> Self {
        self.div_trunc(rhs)
    }

    /// Checked floor division.
    #[must_use]
    pub fn checked_div_floor(&self, rhs: &Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Floor modulus (for unsigned integers, identical to `%`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    pub fn mod_floor(&self, rhs: &Self) -> Self {
        self.rem_trunc(rhs)
    }

    /// Checked floor modulus.
    #[must_use]
    pub fn checked_mod_floor(&self, rhs: &Self) -> Option<Self> {
        self.checked_rem(rhs)
    }

    /// Ceiling division. Rounds quotient toward positive infinity.
    ///
    /// # Panics
    /// Panics if `rhs` is zero.
    #[must_use]
    #[track_caller]
    pub fn div_ceil(&self, rhs: &Self) -> Self {
        assert!(!rhs.value.is_zero(), "Division by zero");
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.div_ceil(&rhs.value),
            precision,
        };
        // For unsigned `a` and `b >= 1`, `ceil(a / b) <= a`; consequently the
        // result fits the left precision and thus the no-narrower combined one.
        result.debug_assert_valid();
        result
    }

    /// Checked ceiling division. Returns `None` if `rhs` is zero.
    #[must_use]
    pub fn checked_div_ceil(&self, rhs: &Self) -> Option<Self> {
        if rhs.value.is_zero() {
            return None;
        }
        let precision = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.div_ceil(&rhs.value),
            precision,
        };
        // The same ceiling bound proved by `div_ceil` makes overflow
        // impossible after the divisor check above.
        result.debug_assert_valid();
        Some(result)
    }
}
