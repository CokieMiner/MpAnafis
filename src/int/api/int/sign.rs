//! Sign manipulation: the operations that read or rewrite the sign without
//! touching the magnitude.
//!
//! `MpInt` stores a magnitude and a sign flag, so each of these is a flag
//! write plus one precision check. The check is the reason they are not free:
//! under a bounded precision the minimum value has no representable negation,
//! and every method here has to decide whether to panic ([`MpInt::abs`],
//! [`MpInt::abs_assign`]) or report ([`MpInt::checked_abs`]).

use core::cmp::Ordering;

use super::{InternalMpInt, InternalMpUint, MpInt};

impl MpInt {
    /// Returns the absolute value.
    ///
    /// # Panics
    /// Panics if `self` is the minimum signed value for a bounded precision
    /// (for example, `-128` for `Bounded(8)`), because its absolute value would
    /// overflow the same precision.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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
        if self.is_bounded_minimum() {
            return None;
        }
        Some(self.abs())
    }

    /// Computes the positive difference between `self` and `other`.
    /// Equivalent to `(self - other).max(0)`.
    ///
    /// This is `num_traits::Signed::abs_sub`, which is *not* the absolute
    /// difference: it clamps at zero rather than taking a magnitude. Use
    /// [`MpInt::abs_diff`] for `|self - other|`.
    ///
    /// # Panics
    /// Panics if `checked_sub` evaluates an underflow when `self > other` (for example, precision mismatch limits).
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
    pub fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            let result = Self {
                value: InternalMpInt::zero(),
                precision: self.precision.combine_for_binary_op(other.precision),
            };
            result.debug_assert_valid();
            result
        } else {
            self.checked_sub(other)
                .expect("abs_sub: self > other guarantees no underflow")
        }
    }

    /// Returns -1, 0, or 1 indicating the sign of this value.
    #[must_use]
    #[allow(
        clippy::same_name_method,
        reason = "Provided as an inherent method for convenience without needing to import num_traits"
    )]
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

    /// Returns whether the value is the most negative one its bounded precision
    /// can hold, which is the only value whose negation does not fit.
    ///
    /// Always `false` under unlimited precision, where every negation fits.
    fn is_bounded_minimum(&self) -> bool {
        let Some(bits) = self.precision.significant_bits() else {
            return false;
        };
        let min_magnitude = InternalMpUint::one().shl(bits.saturating_sub(1));
        !self.value.is_positive && self.value.abs.cmp(&min_magnitude) == Ordering::Equal
    }

    /// Panics when negating `self` would overflow its bounded precision.
    #[track_caller]
    fn assert_negation_fits(&self, operation: &str) {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                !self.is_bounded_minimum(),
                "MpInt {operation} overflow for Bounded({bits})"
            );
        }
    }
}
