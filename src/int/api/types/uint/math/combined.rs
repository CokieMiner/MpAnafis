//! Combined unsigned arithmetic APIs.

use crate::error::MpError;

use super::{InternalMpUint, MpUint, Precision};

impl MpUint {
    /// Direct shift-multiplication by `2^n`: computes `self * 2^n`.
    ///
    /// # Panics
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn mul_2exp(&self, shift: usize) -> Self {
        if let Some(bits) = self.precision.significant_bits() {
            assert!(
                !self.value.bounded_shl_overflows(bits, shift),
                "MpUint mul_2exp overflow for Bounded({bits})"
            );
        }
        let result = Self {
            value: self.value.shl(shift),
            precision: self.precision,
        };
        // The exact shift bound establishes fit before constructing the result.
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

    /// Returns `(lower, upper)` such that `self * other = lower + upper * 2^w`.
    ///
    /// The word width `w` is the combined bounded precision. With unlimited
    /// precision, `lower` is the exact product and `upper` is zero.
    #[must_use]
    pub fn widening_mul(&self, other: &Self) -> (Self, Self) {
        let p = self.precision.combine_for_binary_op(other.precision);
        split_wide_value(self.value.mul(&other.value), p)
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
        Ok(split_bounded_value(self.value.mul(&other.value), p, bits))
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
        split_wide_value(prod, p)
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
        Ok(split_bounded_value(prod, p, bits))
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
        split_wide_value(prod, p)
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
        let mut sum = self.value.add(&other.value);
        sum.shr_assign(1);
        let res = Self {
            value: sum,
            precision: p,
        };
        res.debug_assert_valid();
        res
    }
}

fn split_wide_value(value: InternalMpUint, precision: Precision) -> (MpUint, MpUint) {
    if let Some(bits) = precision.significant_bits() {
        split_bounded_value(value, precision, bits)
    } else {
        (
            MpUint { value, precision },
            MpUint {
                value: InternalMpUint::zero(),
                precision,
            },
        )
    }
}

fn split_bounded_value(
    value: InternalMpUint,
    precision: Precision,
    bits: usize,
) -> (MpUint, MpUint) {
    let upper = MpUint {
        value: value.shr(bits),
        precision,
    };
    let lower = MpUint {
        value: value.apply_wrapping(bits),
        precision,
    };
    lower.debug_assert_valid();
    upper.debug_assert_valid();
    (lower, upper)
}
