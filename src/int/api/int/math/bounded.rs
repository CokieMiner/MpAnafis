//! Wrapping, overflowing, and saturating signed arithmetic APIs.

use super::{ArbiInt, ArbiUint, InternalArbiInt, Precision};

impl ArbiInt {
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
    #[must_use]
    pub fn wrapping_sub(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let diff = self.value.sub(&rhs.value);
        let value = if let Some(bits) = p.significant_bits() {
            diff.apply_wrapping(bits)
        } else {
            diff
        };
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
        let value = if rhs.value.abs.is_zero() {
            InternalArbiInt::zero()
        } else if p
            .significant_bits()
            .is_some_and(|bits| self.value.bounded_division_overflows(&rhs.value, bits))
        {
            // `MIN_bits / -1 = 2^(bits - 1)`, whose `bits`-wide wrapping value
            // is `MIN_bits` itself. Avoid both magnitude division and a
            // sign-magnitude -> two's-complement -> sign-magnitude round trip.
            self.value.clone()
        } else {
            self.value.div(&rhs.value)
        };
        // Apart from `MIN / -1`, `|self / rhs| <= |self|`, so no wrapping work
        // is required after the division.
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
        let value = if rhs.value.abs.is_zero() {
            InternalArbiInt::zero()
        } else {
            self.value.rem(&rhs.value)
        };
        // `|self % rhs| <= |self|`; this includes `MIN % -1 = 0`, so a valid
        // remainder never needs width reduction.
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
            let over = sum.required_signed_bits_for_bounded_storage() > bits;
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

    /// Overflowing subtraction. Returns the result and a boolean indicating
    /// whether overflow occurred.
    #[must_use]
    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let diff = self.value.sub(&rhs.value);
        let (overflow, value) = if let Some(bits) = p.significant_bits() {
            let over = diff.required_signed_bits_for_bounded_storage() > bits;
            (over, diff.apply_wrapping(bits))
        } else {
            (false, diff)
        };
        let result = Self {
            value,
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
            let over = prod.required_signed_bits_for_bounded_storage() > bits;
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
        if rhs.value.abs.is_zero() {
            let result = Self {
                value: InternalArbiInt::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return (result, true);
        }
        let overflow = p
            .significant_bits()
            .is_some_and(|bits| self.value.bounded_division_overflows(&rhs.value, bits));
        let value = if overflow {
            // The wrapped result of the sole overflow case, `MIN / -1`, is MIN.
            self.value.clone()
        } else {
            self.value.div(&rhs.value)
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    /// Overflowing remainder. Returns the remainder and a boolean indicating
    /// whether overflow occurred. Division by zero returns (zero, true).
    #[must_use]
    pub fn overflowing_rem(&self, rhs: &Self) -> (Self, bool) {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.abs.is_zero() {
            let result = Self {
                value: InternalArbiInt::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return (result, true);
        }
        let overflow = p
            .significant_bits()
            .is_some_and(|bits| self.value.bounded_division_overflows(&rhs.value, bits));
        let value = if overflow {
            // Rust-compatible overflowing remainder reports the `MIN / -1`
            // overflow even though its wrapped remainder is exactly zero.
            InternalArbiInt::zero()
        } else {
            self.value.rem(&rhs.value)
        };
        let result = Self {
            value,
            precision: p,
        };
        result.debug_assert_valid();
        (result, overflow)
    }

    // ------------------------------------------------------------------
    // saturating_* arithmetic
    // ------------------------------------------------------------------

    /// Saturating addition. Clamps the result to the minimum or maximum
    /// value of the bounded precision.
    #[must_use]
    pub fn saturating_add(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.add(&rhs.value),
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            if result.value.is_positive {
                Self::max_for_precision(bits)
            } else {
                Self::min_for_precision(bits)
            }
        } else {
            result.debug_assert_valid();
            result
        }
    }

    /// Saturating subtraction. Clamps the result to the minimum or maximum
    /// value of the bounded precision.
    #[must_use]
    pub fn saturating_sub(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.sub(&rhs.value),
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            if result.value.is_positive {
                Self::max_for_precision(bits)
            } else {
                Self::min_for_precision(bits)
            }
        } else {
            result.debug_assert_valid();
            result
        }
    }

    /// Computes the absolute difference between `self` and `other`.
    #[must_use]
    pub fn abs_diff(&self, other: &Self) -> ArbiUint {
        let diff_val = if self.is_negative() == other.is_negative() {
            let (larger, smaller) = if self.value.abs >= other.value.abs {
                (&self.value.abs, &other.value.abs)
            } else {
                (&other.value.abs, &self.value.abs)
            };
            larger.sub(smaller)
        } else {
            self.value.abs.add(&other.value.abs)
        };

        let p = match (self.precision, other.precision) {
            (Precision::Bounded(a), Precision::Bounded(b)) => {
                let max_bits = a.get().max(b.get());
                Precision::new_bounded(max_bits.saturating_add(1)).unwrap_or(Precision::Unlimited)
            }
            _ => Precision::Unlimited,
        };

        let result = ArbiUint {
            value: diff_val,
            precision: p,
        };
        result.debug_assert_valid();
        result
    }

    /// Saturating multiplication. Clamps the result to the minimum or maximum
    /// value of the bounded precision.
    #[must_use]
    pub fn saturating_mul(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        let result = Self {
            value: self.value.mul(&rhs.value),
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            if result.value.is_positive {
                Self::max_for_precision(bits)
            } else {
                Self::min_for_precision(bits)
            }
        } else {
            result.debug_assert_valid();
            result
        }
    }

    /// Saturating division. Returns zero on division by zero.
    #[must_use]
    pub fn saturating_div(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.abs.is_zero() {
            let result = Self {
                value: InternalArbiInt::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return result;
        }
        if let Some(bits) = p.significant_bits()
            && self.value.bounded_division_overflows(&rhs.value, bits)
        {
            return Self::max_for_precision(bits);
        }
        let result = Self {
            value: self.value.div(&rhs.value),
            precision: p,
        };
        // `MIN / -1` is the sole signed division overflow and was handled
        // above; every other quotient has magnitude at most `self`.
        result.debug_assert_valid();
        result
    }

    /// Saturating remainder. Returns zero on division by zero.
    #[must_use]
    pub fn saturating_rem(&self, rhs: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(rhs.precision);
        if rhs.value.abs.is_zero() {
            let result = Self {
                value: InternalArbiInt::zero(),
                precision: p,
            };
            result.debug_assert_valid();
            return result;
        }
        let result = Self {
            value: self.value.rem(&rhs.value),
            precision: p,
        };
        // `|self % rhs| <= |self|`, so a remainder cannot require saturation.
        result.debug_assert_valid();
        result
    }
}
