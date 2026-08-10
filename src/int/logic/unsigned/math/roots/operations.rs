//! Root entry points and their reusable scratch owners.

use alloc::vec::Vec;

use super::{DivScratch, InternalArbiUint, LIMB_BITS, MulScratch, may_be_square};

/// Growable scratch pool for square root operations. Temps are recycled
/// across recursive calls, avoiding repeated allocations.
#[derive(Debug, Clone, Default)]
pub struct SqrtScratch {
    /// Scratch reused by division steps.
    pub div_scratch: DivScratch,
    /// Scratch reused by multiplication steps.
    pub mul_scratch: MulScratch,
    /// Recyclable integer temporaries.
    pub temps: Vec<InternalArbiUint>,
}

impl SqrtScratch {
    /// Acquires a recycled temp (or a fresh zero if the pool is empty).
    pub fn get_temp(&mut self) -> InternalArbiUint {
        self.temps.pop().unwrap_or_else(InternalArbiUint::zero)
    }

    /// Returns a temp to the pool for reuse.
    pub fn return_temp(&mut self, mut t: InternalArbiUint) {
        t.clear();
        self.temps.push(t);
    }
}

/// Pre-allocated scratch space for `nth_root` Newton iteration.
///
/// Reusing this across calls avoids repeated allocations for the
/// intermediate buffers used inside the Newton loop.
#[derive(Debug, Clone)]
pub struct NthRootScratch {
    /// Scratch for the `x_pow_n_minus_1` intermediate.
    pub x_pow_n_minus_1: InternalArbiUint,
    /// Scratch for the `temp_prod` intermediate.
    pub temp_prod: InternalArbiUint,
    /// Scratch for the `quotient` intermediate.
    pub quotient: InternalArbiUint,
    /// Scratch for the `rem` intermediate.
    pub rem: InternalArbiUint,
    /// Scratch for the `scaled_estimate` intermediate.
    pub scaled_estimate: InternalArbiUint,
    /// Scratch for the `sum` intermediate.
    pub sum: InternalArbiUint,
    /// Scratch for the `next_estimate` intermediate.
    pub next_estimate: InternalArbiUint,
    /// Scratch for the `base_pow` intermediate.
    pub base_pow: InternalArbiUint,
    /// Scratch for division.
    pub div_scratch: DivScratch,
    /// Scratch for multiplication.
    pub mul_scratch: MulScratch,
}

impl Default for NthRootScratch {
    fn default() -> Self {
        Self {
            x_pow_n_minus_1: InternalArbiUint::zero(),
            temp_prod: InternalArbiUint::zero(),
            quotient: InternalArbiUint::zero(),
            rem: InternalArbiUint::zero(),
            scaled_estimate: InternalArbiUint::zero(),
            sum: InternalArbiUint::zero(),
            next_estimate: InternalArbiUint::zero(),
            base_pow: InternalArbiUint::zero(),
            div_scratch: DivScratch::default(),
            mul_scratch: MulScratch::default(),
        }
    }
}

impl InternalArbiUint {
    /// Integer square root (floor), i.e. the largest `x` such that
    /// `x^2 <= self`.
    #[must_use]
    pub fn isqrt(&self) -> Self {
        let mut scratch = SqrtScratch::default();
        self.sqrt_with_scratch_mode(&mut scratch, false).0
    }

    /// Computes `(isqrt, remainder)` where `remainder = self - isqrt^2`.
    ///
    /// Uses Zimmermann's subquadratic Karatsuba square root algorithm.
    #[must_use]
    pub fn sqrt_rem(&self) -> (Self, Self) {
        let mut scratch = SqrtScratch::default();
        self.sqrt_with_scratch_mode(&mut scratch, true)
    }

    fn sqrt_with_scratch_mode(
        &self,
        scratch: &mut SqrtScratch,
        need_remainder: bool,
    ) -> (Self, Self) {
        let len = self.limbs().len();
        if len <= 2 {
            if need_remainder {
                return scratch.sqrt_rem_basecase(self);
            }
            return (scratch.isqrt_basecase(self), Self::zero());
        }

        let k = len.div_ceil(4);
        let target_bits = k.wrapping_mul(LIMB_BITS * 4);
        let actual_bits = self.significant_bits();
        let mut shift_bits = target_bits.wrapping_sub(actual_bits);
        if shift_bits & 1 == 1 {
            shift_bits = shift_bits.wrapping_sub(1);
        }

        let mut n = self.clone();
        if shift_bits > 0 {
            n.shl_assign(shift_bits);
        }

        let (s, r) = scratch.sqrt_rem_recursive(&n, need_remainder);

        if shift_bits == 0 {
            return (s, r);
        }

        let sh = shift_bits >> 1;

        let mut s_ret = s.clone();
        s_ret.shr_assign(sh);

        if !need_remainder {
            return (s_ret, Self::zero());
        }

        let mut s_shifted = scratch.get_temp();
        s_shifted.clone_from(&s_ret);
        s_shifted.shl_assign(sh);

        let mut low_bits = scratch.get_temp();
        low_bits.clone_from(&s);
        low_bits.sub_assign(&s_shifted);
        scratch.return_temp(s_shifted);

        let mut two_s_minus_low = s;
        two_s_minus_low.shl_assign(1);
        two_s_minus_low.sub_assign(&low_bits);

        let mut rem = scratch.get_temp();
        rem.assign_product_with_scratch(&low_bits, &two_s_minus_low, &mut scratch.mul_scratch);
        scratch.return_temp(low_bits);
        rem.add_assign(&r);
        rem.shr_assign(shift_bits);

        (s_ret, rem)
    }

    /// Integer `n`th root (floor), i.e. the largest `x` such that
    /// `x^n <= self`.
    ///
    /// `n` must be >= 1.  `nth_root(0) = 0` for any `n`.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Self {
        debug_assert!(n > 0, "internal nth root requires a positive degree");
        if self.is_zero() || self.is_one() || n == 1 {
            return self.clone();
        }
        if n == 2 {
            return self.isqrt();
        }
        let mut scratch = NthRootScratch::default();
        if self.limbs().len() == 1 {
            return NthRootScratch::nth_root_single_limb(self, n);
        }

        scratch.nth_root_multi_limb(self, n)
    }

    /// Returns `true` when `self` is a perfect square.
    ///
    /// The residue screen in [`screen`] rejects 99.79% of non-squares outright,
    /// so the square root below only runs for the small fraction that survives.
    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        if !may_be_square(self) {
            return false;
        }
        self.sqrt_rem().1.is_zero()
    }
}
