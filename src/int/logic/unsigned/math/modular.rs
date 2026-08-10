//! Modular arithmetic for [`InternalArbiUint`].

use core::{cmp::Ordering, mem::replace};

use super::{
    BarrettDomain, BarrettScratch, DivScratch, Division, InternalArbiUint, MontgomeryDomain,
    MulScratch,
};

/// Reusable Montgomery domain for repeated operations with one odd modulus.
#[derive(Debug, Clone)]
pub struct MontgomeryModDomain {
    domain: MontgomeryDomain,
    div_scratch: DivScratch,
    mul_scratch: MulScratch,
    temp_prod: InternalArbiUint,
    reduced_a: InternalArbiUint,
    reduced_b: InternalArbiUint,
}

impl MontgomeryModDomain {
    /// Creates a reusable Montgomery domain.
    #[must_use]
    pub fn new(modulus: &InternalArbiUint) -> Self {
        Self {
            domain: MontgomeryDomain::new(modulus),
            div_scratch: DivScratch::default(),
            mul_scratch: MulScratch::default(),
            temp_prod: InternalArbiUint::zero(),
            reduced_a: InternalArbiUint::zero(),
            reduced_b: InternalArbiUint::zero(),
        }
    }

    /// Performs Montgomery multiplication `(a * b * R^-1) mod modulus`.
    ///
    /// Inputs greater than or equal to the modulus are reduced before the
    /// Montgomery product, matching [`InternalArbiUint::montgomery_mul`].
    #[must_use]
    pub fn mul(&mut self, a: &InternalArbiUint, b: &InternalArbiUint) -> InternalArbiUint {
        let op_a = reduce_operand_for_domain(
            a,
            &self.domain.modulus,
            &mut self.reduced_a,
            &mut self.div_scratch,
        );
        let op_b = reduce_operand_for_domain(
            b,
            &self.domain.modulus,
            &mut self.reduced_b,
            &mut self.div_scratch,
        );
        let mut out = InternalArbiUint::zero();
        self.domain.mul_into_with_scratch(
            op_a,
            op_b,
            &mut out,
            &mut self.temp_prod,
            &mut self.mul_scratch,
        );
        out
    }
}

/// Reusable Barrett domain for repeated reductions with one modulus.
#[derive(Debug, Clone)]
pub struct BarrettModDomain {
    domain: BarrettDomain,
    mul_scratch: MulScratch,
    barrett_scratch: BarrettScratch,
}

impl BarrettModDomain {
    /// Creates a reusable Barrett domain.
    #[must_use]
    pub fn new(modulus: &InternalArbiUint) -> Self {
        Self {
            domain: BarrettDomain::new(modulus),
            mul_scratch: MulScratch::default(),
            barrett_scratch: BarrettScratch::default(),
        }
    }

    /// Reduces `value` modulo this domain's modulus.
    pub fn reduce_into(&mut self, value: &InternalArbiUint, out: &mut InternalArbiUint) {
        self.domain.reduce_into_with_barrett_scratch(
            value,
            out,
            &mut self.mul_scratch,
            &mut self.barrett_scratch,
        );
    }
}

impl InternalArbiUint {
    /// Computes `(self + other) % modulus`.
    #[must_use]
    pub fn add_mod(&self, other: &Self, modulus: &Self) -> Self {
        let mut out = Self::zero();
        self.add_mod_into(other, modulus, &mut out);
        out
    }

    /// Computes `(self + other) % modulus` into `out`.
    ///
    #[allow(clippy::inline_always, reason = "Hot path in modular arithmetic loops")]
    #[inline(always)]
    pub fn add_mod_into(&self, other: &Self, modulus: &Self, out: &mut Self) {
        debug_assert!(
            !modulus.is_zero(),
            "modular addition requires a non-zero modulus"
        );
        // Fused: compute self + other directly into `out` without intermediate clone.
        out.assign_sum(self, other);
        if (*out).cmp(modulus) != Ordering::Less {
            // Fast path: when inputs < modulus, sum < 2*modulus, one subtraction suffices
            out.sub_assign(modulus);
            if (*out).cmp(modulus) != Ordering::Less {
                // Result still >= modulus — rare, use full division
                let mut rem = Self::zero();
                let mut scratch = DivScratch::default();
                Division::rem_into(out, modulus, &mut rem, &mut scratch);
                out.clone_from(&rem);
            }
        }
    }

    /// Computes `(self - other) % modulus`.
    ///
    /// The result is always in `[0, modulus)`.
    #[must_use]
    pub fn sub_mod(&self, other: &Self, modulus: &Self) -> Self {
        let mut out = Self::zero();
        self.sub_mod_into(other, modulus, &mut out);
        out
    }

    /// Computes `(self - other) % modulus` into `out`.
    ///
    /// The result is always in `[0, modulus)`.
    #[allow(clippy::inline_always, reason = "Hot path in modular arithmetic loops")]
    #[inline(always)]
    pub fn sub_mod_into(&self, other: &Self, modulus: &Self, out: &mut Self) {
        debug_assert!(
            !modulus.is_zero(),
            "modular subtraction requires a non-zero modulus"
        );
        if self.cmp(other) == Ordering::Less {
            // Fused: compute other - self directly into `out`.
            let underflowed = out.assign_difference(other, self);
            // `other >= self` by the cmp above, so underflowed is always false.
            debug_assert!(!underflowed, "underflowed on fused subtract");
            if (*out).cmp(modulus) != Ordering::Less {
                // Fast path: when other - self < 2*modulus, at most one subtraction
                // suffices. Check if out < 2*modulus by comparing out - modulus < modulus.
                out.sub_assign(modulus);
                if (*out).cmp(modulus) != Ordering::Less {
                    // Result still >= modulus — rare, use full division
                    let mut rem = Self::zero();
                    let mut scratch = DivScratch::default();
                    Division::rem_into(out, modulus, &mut rem, &mut scratch);
                    if rem.is_zero() {
                        out.clear();
                        return;
                    }
                    // Fused: compute modulus - rem directly into `out`.
                    let inner_underflowed = out.assign_difference(modulus, &rem);
                    debug_assert!(!inner_underflowed, "underflowed on mod subtract");
                    return;
                }
                // After one subtraction, out < modulus. If out == 0, it's already correct.
                if out.is_zero() {
                    return;
                }
                // out is in (0, modulus), but we need modulus - out.
                // Fused: compute modulus - out directly into `out`.
                // Use a temporary to hold the current out value since `out` is both src and dst.
                let current_out = out.clone();
                let inner_underflowed2 = out.assign_difference(modulus, &current_out);
                debug_assert!(!inner_underflowed2, "underflowed on mod subtract");
            } else if out.is_zero() {
                // Do nothing, out is already 0
            } else {
                // out < modulus and out != 0, so result is modulus - out.
                // Fused: compute modulus - out directly into `out`.
                let current_out = out.clone();
                let inner_underflowed3 = out.assign_difference(modulus, &current_out);
                debug_assert!(!inner_underflowed3, "underflowed on mod subtract");
            }
        } else {
            // Fused: compute self - other directly into `out`.
            let underflowed = out.assign_difference(self, other);
            // `self >= other` by the cmp above, so underflowed is always false.
            debug_assert!(!underflowed, "underflowed on fused subtract");
            if (*out).cmp(modulus) != Ordering::Less {
                // Fast path: when self - other < 2*modulus, one subtraction suffices.
                out.sub_assign(modulus);
                if (*out).cmp(modulus) != Ordering::Less {
                    // Result still >= modulus — rare, use full division
                    let mut rem = Self::zero();
                    let mut scratch = DivScratch::default();
                    Division::rem_into(out, modulus, &mut rem, &mut scratch);
                    out.clone_from(&rem);
                }
            }
        }
    }

    /// Computes `(self * other) % modulus`.
    ///
    /// For repeated modular multiplications with the same odd modulus,
    /// prefer using `MontgomeryDomain` directly for better performance.
    #[must_use]
    pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Self {
        let mut out = Self::zero();
        self.mul_mod_into(other, modulus, &mut out);
        out
    }

    /// Computes `(self * other) % modulus` into `out`.
    ///
    /// For repeated modular multiplications with the same odd modulus,
    /// prefer using `MontgomeryDomain` directly for better performance.
    #[allow(clippy::inline_always, reason = "Hot path in modular arithmetic loops")]
    #[inline(always)]
    pub fn mul_mod_into(&self, other: &Self, modulus: &Self, out: &mut Self) {
        debug_assert!(
            !modulus.is_zero(),
            "modular multiplication requires a non-zero modulus"
        );
        if modulus.is_one() {
            out.clear();
            return;
        }
        let mut div_scratch = DivScratch::default();
        let mut mul_scratch = MulScratch::default();
        // Use out as the temporary product buffer to avoid an extra product allocation.
        out.assign_product_with_scratch(self, other, &mut mul_scratch);
        let mut rem = replace(&mut div_scratch.mod_rem, Self::zero());
        Division::rem_into(&*out, modulus, &mut rem, &mut div_scratch);
        out.clone_from(&rem);
        div_scratch.mod_rem = rem;
    }

    /// Computes `self^exp mod modulus` using exponentiation by squaring.
    ///
    #[must_use]
    pub fn pow_mod(&self, exp: &Self, modulus: &Self) -> Self {
        debug_assert!(
            !modulus.is_zero(),
            "modular exponentiation requires a non-zero modulus"
        );
        if modulus.is_one() {
            return Self::zero();
        }
        if exp.is_zero() {
            return Self::one();
        }

        let mut scratch = MulScratch::default();
        if modulus.is_odd() {
            return MontgomeryDomain::new(modulus).pow(self, exp, &mut scratch, false);
        }
        BarrettDomain::new(modulus).pow(self, exp, &mut scratch)
    }

    /// Computes the modular inverse of `self` modulo `modulus`.
    ///
    /// Returns `None` when the inverse does not exist (i.e. when
    /// `gcd(self, modulus) != 1`).
    #[must_use]
    pub fn invert(&self, modulus: &Self) -> Option<Self> {
        debug_assert!(
            !modulus.is_zero(),
            "modular inversion requires a non-zero modulus"
        );
        Division::mod_inverse(self, modulus)
    }

    /// Performs Montgomery multiplication `(self * other * R^-1) mod modulus`.
    #[must_use]
    pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Self {
        debug_assert!(
            modulus.is_odd(),
            "Montgomery multiplication requires a non-zero odd modulus"
        );
        MontgomeryModDomain::new(modulus).mul(self, other)
    }

    /// Performs modular reduction `self % modulus` using Barrett reduction.
    #[must_use]
    pub fn barrett_reduce(&self, modulus: &Self) -> Self {
        debug_assert!(
            !modulus.is_zero(),
            "Barrett reduction requires a non-zero modulus"
        );
        if self.limbs().len() > modulus.limbs().len().saturating_mul(2) {
            return self.div_rem(modulus).1;
        }
        let mut domain = BarrettModDomain::new(modulus);
        let mut out = Self::zero();
        domain.reduce_into(self, &mut out);
        out
    }
}

fn reduce_operand_for_domain<'operand>(
    value: &'operand InternalArbiUint,
    modulus: &InternalArbiUint,
    reduced: &'operand mut InternalArbiUint,
    scratch: &mut DivScratch,
) -> &'operand InternalArbiUint {
    if value < modulus {
        return value;
    }
    let mut rem = replace(reduced, InternalArbiUint::zero());
    Division::rem_into(value, modulus, &mut rem, scratch);
    *reduced = rem;
    reduced
}
