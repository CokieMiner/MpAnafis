//! Combined signed arithmetic APIs.

use core::ops::{BitAnd, BitXor, Shl, Shr};

use crate::error::MpError;

use super::{InternalMpInt, MpInt, Precision};

impl MpInt {
    /// Direct shift-multiplication by `2^n`: computes `self * 2^n`.
    ///
    /// # Panics
    /// Panics if the result exceeds bounded precision.
    #[must_use]
    #[track_caller]
    pub fn mul_2exp(&self, shift: usize) -> Self {
        let result = Self {
            value: (&self.value).shl(shift),
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
            value: (&self.value).shr(shift),
            precision: self.precision,
        };
        result.debug_assert_valid();
        result
    }

    /// Double-width signed multiplication returning `(lower, upper)` words.
    ///
    /// For a bounded result precision of `P` bits, each returned value is a
    /// signed `P`-bit word.  Its two's-complement encoding is the corresponding
    /// half of the exact product: reconstruct the `2P`-bit result by encoding
    /// both words in `P` bits, shifting the upper encoding by `P`, adding the
    /// lower encoding, and decoding the combined bits as signed.  In
    /// particular, do not reconstruct with signed arithmetic on `lower` when
    /// it is negative; its low-word encoding is then the wrapped `P`-bit value.
    ///
    /// If either operand is unbounded, no finite word width exists.  In that
    /// case this method returns the exact product as `lower` and zero as
    /// `upper`; use [`Self::try_widening_mul`] when a bounded word split is
    /// required.
    #[must_use]
    pub fn widening_mul(&self, other: &Self) -> (Self, Self) {
        let p = self.precision.combine_for_binary_op(other.precision);
        split_wide_value(self.value.mul(&other.value), p)
    }

    /// Fallible double-width signed multiplication returning `Result<(lower, upper), MpError>`.
    ///
    /// For bounded precision, the two words follow [`Self::widening_mul`]'s
    /// reconstructable two's-complement word contract.  Unlike the infallible
    /// method, this returns [`MpError::WidthRequired`] for unbounded operands
    /// because an unbounded value has no finite word width.
    ///
    /// # Errors
    /// Returns [`MpError::WidthRequired`] when called on unbounded [`MpInt`].
    pub fn try_widening_mul(&self, other: &Self) -> Result<(Self, Self), MpError> {
        let p = self.precision.combine_for_binary_op(other.precision);
        let Some(bits) = p.significant_bits() else {
            return Err(MpError::WidthRequired);
        };
        Ok(split_bounded_value(self.value.mul(&other.value), p, bits))
    }

    /// Double-width signed multiplication with an additive carry parameter, returning `(lower, upper)`.
    ///
    /// The carry is an exact signed addend before splitting.  For bounded
    /// precision, the returned words use the same reconstructable
    /// two's-complement contract as [`Self::widening_mul`].  For unbounded
    /// precision, the exact sum is returned as `lower` and `upper` is zero.
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

    /// Fallible double-width signed carrying multiplication returning `Result<(lower, upper), MpError>`.
    ///
    /// The carry is added exactly before the bounded two's-complement word
    /// split.  Unbounded operands return [`MpError::WidthRequired`], because
    /// no finite word width can describe the pair.
    ///
    /// # Errors
    /// Returns [`MpError::WidthRequired`] when called on unbounded [`MpInt`].
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

    /// Double-width signed multiply-accumulate with two additive carry terms.
    ///
    /// Both carries are added exactly before splitting.  For bounded
    /// precision, `(lower, upper)` follows [`Self::widening_mul`]'s signed
    /// two's-complement word contract; for unbounded precision, the exact
    /// result is in `lower` and `upper` is zero.
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
    /// For odd sums, rounding follows Rust primitive integer midpoint semantics (toward negative infinity / floor).
    #[must_use]
    pub fn midpoint(&self, other: &Self) -> Self {
        let p = self.precision.combine_for_binary_op(other.precision);
        let and = (&self.value).bitand(&other.value);
        let xor_half = (&self.value).bitxor(&other.value).shr(1_usize);
        let res = Self {
            value: and.add(&xor_half),
            precision: p,
        };
        res.debug_assert_valid();
        res
    }
}

fn split_wide_value(value: InternalMpInt, precision: Precision) -> (MpInt, MpInt) {
    if let Some(bits) = precision.significant_bits() {
        split_bounded_value(value, precision, bits)
    } else {
        (
            MpInt { value, precision },
            MpInt {
                value: InternalMpInt::zero(),
                precision,
            },
        )
    }
}

fn split_bounded_value(value: InternalMpInt, precision: Precision, bits: usize) -> (MpInt, MpInt) {
    let upper_value = (&value).shr(bits);
    let lower_value = value.apply_wrapping(bits);
    let lower = MpInt {
        value: lower_value,
        precision,
    };
    let upper = MpInt {
        value: upper_value,
        precision,
    };
    lower.debug_assert_valid();
    upper.debug_assert_valid();
    (lower, upper)
}
