//! Absolute values, positive differences, and sign normalization.
//!
//! The bounded minimum, `-2^(w-1)`, has no positive counterpart at width `w`.
//! Absolute-value operations validate this endpoint before changing the sign.

#![cfg_attr(
    feature = "num-traits",
    expect(
        clippy::same_name_method,
        reason = "Inherent sign operations intentionally mirror num_traits conveniences"
    )
)]

use super::{InternalMpInt, InternalMpUint, MpInt};

impl MpInt {
    /// Returns the absolute value.
    ///
    /// # Panics
    /// Panics if `self` is the minimum signed value for a bounded precision
    /// (for example, `-128` for `Bounded(8)`), because its absolute value would
    /// overflow the same precision.
    #[must_use]
    pub fn abs(&self) -> Self {
        self.assert_negation_fits("abs");
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.clone(),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Sets the value to its absolute value in-place.
    ///
    /// # Panics
    /// Panics if `self` is the minimum signed value for a bounded precision
    /// because its absolute value would overflow that precision.
    pub fn abs_assign(&mut self) {
        self.assert_negation_fits("abs");
        self.value.is_positive = true;
        self.debug_assert_valid();
    }

    /// Checked absolute value. Returns `None` if `self` is the minimum
    /// bounded value (for example, -128 for `Bounded(8)`).
    #[must_use]
    pub fn checked_abs(&self) -> Option<Self> {
        if self
            .precision
            .significant_bits()
            .is_some_and(|bits| self.value.is_signed_min_for_width(bits))
        {
            return None;
        }
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.clone(),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Computes the positive difference between `self` and `other`.
    /// Equivalent to `(self - other).max(0)`.
    ///
    /// Implements the positive difference used by `num_traits::Signed::abs_sub`.
    /// [`MpInt::abs_diff`] computes `|self - other|`.
    ///
    /// # Panics
    /// Panics if the positive difference exceeds the operands' combined
    /// bounded precision.
    #[must_use]
    pub fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            let result = Self {
                value: InternalMpInt::zero(),
                precision: self.precision.combine_for_binary_op(other.precision),
            };
            result.debug_assert_valid();
            result
        } else {
            // self > other gives a positive difference. Only bounded overflow
            // can fail after the magnitude subtraction.
            let value = self.value.sub(&other.value);
            let precision = self.precision.combine_for_binary_op(other.precision);
            let result = Self { value, precision };
            result.assert_fits("abs_sub");
            result.debug_assert_valid();
            result
        }
    }

    /// Returns -1, 0, or 1 indicating the sign of this value.
    #[must_use]
    pub fn signum(&self) -> Self {
        let result = if self.value.abs.is_zero() {
            Self {
                value: InternalMpInt::zero(),
                precision: self.precision,
            }
        } else if self.value.is_positive {
            Self {
                value: InternalMpInt::one(),
                precision: self.precision,
            }
        } else {
            Self {
                value: InternalMpInt {
                    abs: InternalMpUint::one(),
                    is_positive: false,
                },
                precision: self.precision,
            }
        };
        result.debug_assert_valid();
        result
    }
}
