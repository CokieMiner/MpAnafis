//! Root entry points and reusable arithmetic workspaces.

use super::{DivScratch, InternalMpUint, MulScratch, Roots};

/// Reusable powers, residues, and quotients for integer root precision growth.
#[derive(Debug, Clone)]
pub struct NthRootScratch {
    /// The denominator x^(n-1).
    pub x_pow_n_minus_1: InternalMpUint,
    /// Product or square under construction.
    pub temp_prod: InternalMpUint,
    /// Division quotient for a root update.
    pub quotient: InternalMpUint,
    /// Shifted prefix residue or Newton difference.
    pub difference: InternalMpUint,
    /// New root digits or Newton decrement after scalar division by n.
    pub correction: InternalMpUint,
    /// Scratch for division.
    pub div_scratch: DivScratch,
    /// Scratch for multiplication.
    pub mul_scratch: MulScratch,
}

impl InternalMpUint {
    /// Returns the greatest integer x such that x^n <= self.
    ///
    /// The caller must validate n >= 1. Zero has root zero for every degree.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Self {
        debug_assert!(n > 0, "internal nth root requires a positive degree");
        if self.is_zero() || self.is_one() || n == 1 {
            return self.clone();
        }
        if n == 2 {
            return self.isqrt();
        }
        if self.limbs().len() == 1 {
            return Roots::nth_root_single_limb(self, n);
        }
        let bits = self.significant_bits();
        NthRootScratch::default().nth_root_multi_limb::<false>(self, n, bits)
    }

    /// Returns whether this value is a perfect square.
    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        if !Roots::may_be_square(self) {
            return false;
        }
        self.sqrt_rem().1.is_zero()
    }

    /// Returns the greatest integer x such that x^2 <= self.
    #[must_use]
    pub fn isqrt(&self) -> Self {
        Roots::sqrt::<false>(self).0
    }

    /// Returns (floor(sqrt(self)), self-floor(sqrt(self))^2).
    #[must_use]
    pub fn sqrt_rem(&self) -> (Self, Self) {
        Roots::sqrt::<true>(self)
    }
}

impl Default for NthRootScratch {
    fn default() -> Self {
        Self {
            x_pow_n_minus_1: InternalMpUint::zero(),
            temp_prod: InternalMpUint::zero(),
            quotient: InternalMpUint::zero(),
            difference: InternalMpUint::zero(),
            correction: InternalMpUint::zero(),
            div_scratch: DivScratch::default(),
            mul_scratch: MulScratch::default(),
        }
    }
}
