//! Unsigned integer property and comparison helper APIs.

use super::MpUint;

impl MpUint {
    /// Returns `true` if this integer is zero.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Returns `true` if this integer is one.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Returns `true` if this integer is even.
    #[inline]
    #[must_use]
    pub fn is_even(&self) -> bool {
        self.value.is_even()
    }

    /// Returns `true` if this integer is odd.
    #[inline]
    #[must_use]
    pub fn is_odd(&self) -> bool {
        self.value.is_odd()
    }

    /// Returns the number of significant bits in the magnitude of this integer.
    /// Returns `0` if the value is zero.
    #[inline]
    #[must_use]
    pub fn significant_bits(&self) -> usize {
        self.value.significant_bits()
    }

    /// Returns the smaller of two unsigned integers.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn min(self, other: Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let mut result = if self.value < other.value {
            self
        } else {
            other
        };
        result.precision = p;
        result.debug_assert_valid();
        result
    }

    /// Returns the larger of two unsigned integers.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn max(self, other: Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let mut result = if self.value > other.value {
            self
        } else {
            other
        };
        result.precision = p;
        result.debug_assert_valid();
        result
    }

    /// Clamps `self` within the inclusive range `[min, max]`.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        let p = self
            .precision
            .combine_for_binary_op(min.precision)
            .combine_for_binary_op(max.precision);
        let mut result = if self.value < min.value {
            min
        } else if self.value > max.value {
            max
        } else {
            self
        };
        result.precision = p;
        result.debug_assert_valid();
        result
    }
}
