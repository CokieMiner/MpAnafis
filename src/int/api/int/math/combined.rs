//! Combined signed arithmetic APIs.

use core::ops::{BitAnd, BitXor, Shr};

use super::MpInt;

impl MpInt {
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
