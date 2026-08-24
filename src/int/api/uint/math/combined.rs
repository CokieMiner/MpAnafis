//! Combined unsigned arithmetic APIs.

use crate::error::MpError;

use super::{InternalMpUint, MpUint};

impl MpUint {
    /// Direct shift-multiplication by `2^n`: computes `self * 2^n`.
    ///
    /// # Panics
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn mul_2exp(&self, shift: usize) -> Self {
        let result = Self {
            value: self.value.shl(shift),
            precision: self.precision,
        };
        result.assert_fits("mul_2exp");
        result.debug_assert_valid();
        result
    }

    /// Direct power-of-two division: computes `self / 2^n` (equivalent to `self >> n`).
    #[must_use]
    pub fn div_2exp(&self, shift: usize) -> Self {
        let result = Self {
            value: self.value.shr(shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Double-width multiplication returning `(lower, upper)` words.
    #[must_use]
    pub fn widening_mul(&self, other: &Self) -> (Self, Self) {
        let p = self.precision.combine_for_binary_op(other.precision);
        let prod = self.value.mul(&other.value);
        if let Some(bits) = p.significant_bits() {
            let lower_val = prod.clone().apply_wrapping(bits);
            let upper_val = prod.shr(bits);
            let lower = Self {
                value: lower_val,
                precision: p,
            };
            let upper = Self {
                value: upper_val,
                precision: p,
            };
            lower.debug_assert_valid();
            upper.debug_assert_valid();
            (lower, upper)
        } else {
            let lower = Self {
                value: prod,
                precision: p,
            };
            let upper = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            (lower, upper)
        }
    }

    /// Fallible double-width multiplication returning `Result<(lower, upper), MpError>`.
    ///
    /// # Errors
    /// Returns [`MpError::WidthRequired`] when called on unbounded [`MpUint`].
    pub fn try_widening_mul(&self, other: &Self) -> Result<(Self, Self), MpError> {
        let p = self.precision.combine_for_binary_op(other.precision);
        let Some(bits) = p.significant_bits() else {
            return Err(MpError::WidthRequired);
        };
        let prod = self.value.mul(&other.value);
        let lower_val = prod.clone().apply_wrapping(bits);
        let upper_val = prod.shr(bits);
        let lower = Self {
            value: lower_val,
            precision: p,
        };
        let upper = Self {
            value: upper_val,
            precision: p,
        };
        lower.debug_assert_valid();
        upper.debug_assert_valid();
        Ok((lower, upper))
    }

    /// Double-width multiplication with an additive carry parameter, returning `(lower, upper)`.
    #[must_use]
    pub fn carrying_mul(&self, other: &Self, carry: &Self) -> (Self, Self) {
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(carry.precision);
        let mut prod = self.value.mul(&other.value);
        prod.add_assign(&carry.value);
        if let Some(bits) = p.significant_bits() {
            let lower_val = prod.clone().apply_wrapping(bits);
            let upper_val = prod.shr(bits);
            let lower = Self {
                value: lower_val,
                precision: p,
            };
            let upper = Self {
                value: upper_val,
                precision: p,
            };
            lower.debug_assert_valid();
            upper.debug_assert_valid();
            (lower, upper)
        } else {
            let lower = Self {
                value: prod,
                precision: p,
            };
            let upper = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            (lower, upper)
        }
    }

    /// Fallible double-width carrying multiplication returning `Result<(lower, upper), MpError>`.
    ///
    /// # Errors
    /// Returns [`MpError::WidthRequired`] when called on unbounded [`MpUint`].
    pub fn try_carrying_mul(&self, other: &Self, carry: &Self) -> Result<(Self, Self), MpError> {
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(carry.precision);
        let Some(bits) = p.significant_bits() else {
            return Err(MpError::WidthRequired);
        };
        let mut prod = self.value.mul(&other.value);
        prod.add_assign(&carry.value);
        let lower_val = prod.clone().apply_wrapping(bits);
        let upper_val = prod.shr(bits);
        let lower = Self {
            value: lower_val,
            precision: p,
        };
        let upper = Self {
            value: upper_val,
            precision: p,
        };
        lower.debug_assert_valid();
        upper.debug_assert_valid();
        Ok((lower, upper))
    }

    /// Double-width multiply-accumulate with two additive carry terms.
    #[must_use]
    pub fn carrying_mul_add(&self, other: &Self, carry1: &Self, carry2: &Self) -> (Self, Self) {
        let p = self
            .precision
            .combine_for_binary_op(other.precision)
            .combine_for_binary_op(carry1.precision)
            .combine_for_binary_op(carry2.precision);
        let mut prod = self.value.mul(&other.value);
        prod.add_assign(&carry1.value);
        prod.add_assign(&carry2.value);
        if let Some(bits) = p.significant_bits() {
            let lower_val = prod.clone().apply_wrapping(bits);
            let upper_val = prod.shr(bits);
            let lower = Self {
                value: lower_val,
                precision: p,
            };
            let upper = Self {
                value: upper_val,
                precision: p,
            };
            lower.debug_assert_valid();
            upper.debug_assert_valid();
            (lower, upper)
        } else {
            let lower = Self {
                value: prod,
                precision: p,
            };
            let upper = Self {
                value: InternalMpUint::zero(),
                precision: p,
            };
            (lower, upper)
        }
    }

    /// Fused multiply-add: computes `(self * a) + b` without intermediate precision truncation.
    ///
    /// # Panics
    /// Panics if the exact final result does not fit the operands' combined
    /// bounded precision.
    #[must_use]
    #[track_caller]
    pub fn mul_add(&self, a: &Self, b: &Self) -> Self {
        let p = self
            .precision
            .combine_for_binary_op(a.precision)
            .combine_for_binary_op(b.precision);
        let mut prod = self.value.mul(&a.value);
        prod.add_assign(&b.value);
        let result = Self {
            value: prod,
            precision: p,
        };
        result.assert_fits("fused multiply-add");
        result.debug_assert_valid();
        result
    }

    /// Computes the midpoint `(self + other) / 2` without intermediate precision overflow.
    #[must_use]
    pub fn midpoint(&self, other: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let mut sum = self.value.clone();
        sum.add_assign(&other.value);
        sum.shr_assign(1);
        let res = Self {
            value: sum,
            precision: p,
        };
        res.debug_assert_valid();
        res
    }
}
