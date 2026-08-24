//! Wrapping, overflowing, and saturating unsigned arithmetic APIs.

use super::{InternalMpUint, MpUint};

impl MpUint {
    // wrapping_* arithmetic
    // ------------------------------------------------------------------

    /// Wrapping addition. Truncates the result to the bounded precision.
    #[must_use]
    pub fn wrapping_add(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let sum = self.value.add(&rhs.value);
        let value = if let Some(bits) = p.significant_bits() {
            sum.apply_wrapping(bits)
        } else {
            sum
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Wrapping subtraction. Truncates the result to the bounded precision.
    /// Underflow for bounded unsigned values wraps around.
    ///
    /// # Panics
    /// Panics if unlimited-precision subtraction would underflow. Unlimited
    /// unsigned integers have no finite width around which to wrap.
    #[must_use]
    pub fn wrapping_sub(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let value = p.significant_bits().map_or_else(
            || {
                let (difference, underflowed) = self.value.sub_with_underflow(&rhs.value);
                assert!(
                    !underflowed,
                    "MpUint wrapping_sub for unlimited precision is undefined on underflow; use wrapping_sub_with_width(bits) or saturating_sub"
                );
                difference
            },
            |bits| {
                self.value
                    .wrapping_sub_with_underflow(&rhs.value, bits)
                    .0
            },
        );
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Wrapping multiplication. Truncates the result to the bounded precision.
    #[must_use]
    pub fn wrapping_mul(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let prod = self.value.mul(&rhs.value);
        let value = if let Some(bits) = p.significant_bits() {
            prod.apply_wrapping(bits)
        } else {
            prod
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Wrapping division. Division by zero returns zero.
    #[must_use]
    pub fn wrapping_div(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let value = if rhs.value.is_zero() {
            InternalMpUint::zero()
        } else {
            self.value.div(&rhs.value)
        };
        // For a non-zero divisor, `self / rhs <= self`; the combined precision
        // is at least `self.precision`, so wrapping can never change the quotient.
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Wrapping remainder. Division by zero returns zero.
    #[must_use]
    pub fn wrapping_rem(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let value = if rhs.value.is_zero() {
            InternalMpUint::zero()
        } else {
            self.value.rem(&rhs.value)
        };
        // A remainder is at most `self`; the combined precision is at least
        // `self.precision`, so wrapping can never change it.
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    // ------------------------------------------------------------------
    // overflowing_* arithmetic
    // ------------------------------------------------------------------

    /// Overflowing addition. Returns the result and a boolean indicating
    /// whether overflow occurred.
    #[must_use]
    pub fn overflowing_add(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let sum = self.value.add(&rhs.value);
        let (overflow, value) = if let Some(bits) = p.significant_bits() {
            let over = sum.significant_bits() > bits;
            (over, sum.apply_wrapping(bits))
        } else {
            (false, sum)
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    #[must_use]
    /// Overflowing subtraction. Returns the result and a boolean indicating
    /// whether underflow occurred.
    ///
    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let (diff, overflow) = p.significant_bits().map_or_else(
            || {
                let (difference, underflowed) = self.value.sub_with_underflow(&rhs.value);
                let value = if underflowed {
                    InternalMpUint::zero()
                } else {
                    difference
                };
                (value, underflowed)
            },
            |bits| self.value.wrapping_sub_with_underflow(&rhs.value, bits),
        );
        let result = Self {
            value: diff,
            precision: p,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    /// Overflowing multiplication. Returns the result and a boolean
    /// indicating whether overflow occurred.
    #[must_use]
    pub fn overflowing_mul(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let prod = self.value.mul(&rhs.value);
        let (overflow, value) = if let Some(bits) = p.significant_bits() {
            let over = prod.significant_bits() > bits;
            (over, prod.apply_wrapping(bits))
        } else {
            (false, prod)
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    /// Overflowing division. Returns the quotient and a boolean indicating
    /// whether overflow occurred. Division by zero returns (zero, true).
    #[must_use]
    pub fn overflowing_div(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.is_zero() {
            let result = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return (result, true);
        }
        let value = self.value.div(&rhs.value);
        // For non-zero `rhs`, the quotient is at most `self` and therefore fits
        // the combined precision. Only division by zero sets the overflow flag.
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, false)
    }

    /// Overflowing remainder. Returns the remainder and a boolean indicating
    /// whether overflow occurred. Division by zero returns (zero, true).
    #[must_use]
    pub fn overflowing_rem(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.is_zero() {
            let result = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return (result, true);
        }
        let value = self.value.rem(&rhs.value);
        // A valid remainder is at most `self` and therefore fits the combined
        // precision. Only division by zero sets the overflow flag.
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, false)
    }

    // ------------------------------------------------------------------
    // saturating_* arithmetic
    // ------------------------------------------------------------------

    /// Saturating addition. Clamps the result to the maximum value of the
    /// bounded precision.
    #[must_use]
    pub fn saturating_add(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let sum = self.value.add(&rhs.value);
        if let Some(bits) = p.significant_bits() {
            let max_val = InternalMpUint::max_for_bits(bits);
            let value = if sum.significant_bits() > bits {
                max_val
            } else {
                sum
            };
            let result = Self {
                value,
                precision: p,
            };
            result.debug_assert_valid();
            result
        } else {
            let result = Self {
                value: sum,
                precision: p,
            };
            result.debug_assert_valid();
            result
        }
    }

    /// Saturating subtraction. Clamps the result to zero on underflow.
    #[must_use]
    pub fn saturating_sub(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let (difference, underflowed) = self.value.sub_with_underflow(&rhs.value);
        let value = if underflowed {
            InternalMpUint::zero()
        } else {
            difference
        };
        // A successful difference is at most `self`, so it cannot overflow the
        // combined precision.
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Computes the absolute difference between `self` and `other`.
    #[must_use]
    pub fn abs_diff(&self, other: &Self) -> Self {
        let (larger, smaller) = if self.value >= other.value {
            (&self.value, &other.value)
        } else {
            (&other.value, &self.value)
        };
        let p = self.precision.combine_for_binary_op(other.precision);
        let diff_val = larger.sub(smaller);
        let result = Self {
            value: diff_val,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Saturating multiplication. Clamps the result to the maximum value of
    /// the bounded precision.
    #[must_use]
    pub fn saturating_mul(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let prod = self.value.mul(&rhs.value);
        if let Some(bits) = p.significant_bits() {
            let max_val = InternalMpUint::max_for_bits(bits);
            let value = if prod.significant_bits() > bits {
                max_val
            } else {
                prod
            };
            let result = Self {
                value,
                precision: p,
            };
            result.debug_assert_valid();
            result
        } else {
            let result = Self {
                value: prod,
                precision: p,
            };
            result.debug_assert_valid();
            result
        }
    }

    /// Saturating division. Returns `None`-equivalent (zero) on division by
    /// zero.
    #[must_use]
    pub fn saturating_div(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.is_zero() {
            let result = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return result;
        }
        let result = Self {
            value: self.value.div(&rhs.value),
            precision: p,
        };
        // A quotient is at most `self`, so saturation is impossible here.
        result.debug_assert_valid();
        result
    }

    /// Saturating remainder. Returns zero on division by zero.
    #[must_use]
    pub fn saturating_rem(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.is_zero() {
            let result = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return result;
        }
        let result = Self {
            value: self.value.rem(&rhs.value),
            precision: p,
        };
        // A remainder is at most `self`, so saturation is impossible here.
        result.debug_assert_valid();
        result
    }
}
