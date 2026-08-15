//! Signed integer property and comparison helper APIs.

use super::{InternalMpInt, InternalMpUint, MpInt, MpUint};

impl MpInt {
    /// Returns `true` if this integer is zero.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_zero(&self) -> bool {
        self.value.abs.is_zero()
    }

    /// Returns `true` if the value is positive (greater than zero).
    ///
    /// Zero is neither positive nor negative, so this is not the negation of
    /// [`MpInt::is_negative`].
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_positive(&self) -> bool {
        self.value.is_positive && !self.value.abs.is_zero()
    }

    /// Returns `true` if the value is negative (less than zero).
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_negative(&self) -> bool {
        !self.value.is_positive && !self.value.abs.is_zero()
    }

    /// Returns `true` if the value is exactly -1.
    #[inline]
    #[must_use]
    pub fn is_minus_one(&self) -> bool {
        !self.value.is_positive && self.value.abs.is_one()
    }

    /// Returns `true` if this integer is one.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn is_one(&self) -> bool {
        self.is_positive() && self.value.abs.is_one()
    }

    /// Returns `true` if this integer is even.
    #[inline]
    #[must_use]
    pub fn is_even(&self) -> bool {
        self.value.abs.is_even()
    }

    /// Returns `true` if this integer is odd.
    #[inline]
    #[must_use]
    pub fn is_odd(&self) -> bool {
        self.value.abs.is_odd()
    }

    /// Returns `true` if this integer is a power of two (only valid for positive numbers).
    #[inline]
    #[must_use]
    pub fn is_power_of_two(&self) -> bool {
        self.is_positive() && self.value.abs.count_ones() == 1
    }

    /// Returns the number of significant bits in the magnitude of this integer.
    /// Returns `0` if the value is zero.
    #[inline]
    #[must_use]
    pub fn significant_bits(&self) -> usize {
        self.value.abs.significant_bits()
    }

    /// Returns the unsigned magnitude of the value.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn unsigned_abs(&self) -> MpUint {
        let result = MpUint {
            value: self.value.abs.clone(),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Returns the next power of two greater than or equal to `self`, or `None` if the value is negative or the result cannot be represented.
    #[must_use]
    pub fn checked_next_power_of_two(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        if let Some(width) = self.precision.significant_bits() {
            if self.is_zero() {
                return (width > 1).then(|| {
                    let result = Self {
                        value: InternalMpInt::one(),
                        precision: self.precision,
                    };
                    result.debug_assert_valid();
                    result
                });
            }
            if self.is_power_of_two() {
                return Some(self.clone());
            }
            let next_bit = self.value.abs.significant_bits();
            if next_bit >= width.saturating_sub(1) {
                return None;
            }
            let result = Self {
                value: InternalMpInt {
                    abs: InternalMpUint::one().shl(next_bit),
                    is_positive: true,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            return Some(result);
        }
        if self.is_zero() {
            return Some(Self::one());
        }
        if self.is_power_of_two() {
            return Some(self.clone());
        }
        let bits = self.value.abs.significant_bits();
        let result = Self {
            value: InternalMpInt {
                abs: InternalMpUint::one().shl(bits),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the smaller of two signed integers.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn min(&self, other: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let value = if self < other {
            self.value.clone()
        } else {
            other.value.clone()
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Returns the larger of two signed integers.
    #[inline]
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Inherent method mirrors num_traits trait for ergonomic access without trait import"
    )]
    pub fn max(&self, other: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let value = if self > other {
            self.value.clone()
        } else {
            other.value.clone()
        };
        let result = Self {
            value,
            precision: p,
        };
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
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let p = self
            .precision
            .combine_for_binary_op(min.precision)
            .combine_for_binary_op(max.precision);
        let value = if self < min {
            min.value.clone()
        } else if self > max {
            max.value.clone()
        } else {
            self.value.clone()
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }
}
