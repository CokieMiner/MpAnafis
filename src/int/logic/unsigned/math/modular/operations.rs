//! Modular arithmetic for [`InternalMpUint`].

#![expect(
    unsafe_code,
    reason = "The one-extra-limb input branch establishes an initialized modulus-width high prefix before unchecked slicing."
)]

use core::cmp::Ordering;

use super::{
    BarrettDomain, BarrettScratch, DivScratch, Division, InlineMontgomery, InternalMpUint,
    LimbMontgomery, MONTGOMERY_POW_MOD_THRESHOLD, MontgomeryDomain, MontgomeryScratch, MulScratch,
};

impl InternalMpUint {
    /// Computes `(self + other) % modulus`.
    #[must_use]
    pub fn add_mod(&self, other: &Self, modulus: &Self) -> Self {
        let mut out = Self::zero();
        self.add_mod_into(other, modulus, &mut out);
        out
    }

    /// Computes `(self + other) % modulus` into `out`.
    #[expect(
        clippy::inline_always,
        reason = "Inlining exposes the caller's reduced-operand bound to the fused sum and modulus comparisons."
    )]
    #[inline(always)]
    pub fn add_mod_into(&self, other: &Self, modulus: &Self, out: &mut Self) {
        debug_assert!(
            !modulus.is_zero(),
            "modular addition requires a non-zero modulus"
        );
        // The fused sum reuses out without cloning an operand.
        out.assign_sum(self, other);
        if (*out).cmp(modulus) != Ordering::Less {
            // Reduced operands give sum < 2*modulus, so one subtraction suffices.
            out.sub_assign(modulus);
            if (*out).cmp(modulus) != Ordering::Less {
                // Unreduced operands may require full division.
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
    #[expect(
        clippy::inline_always,
        reason = "Inlining exposes the caller's reduced-operand bound to ordered subtraction and modulus correction."
    )]
    #[inline(always)]
    pub fn sub_mod_into(&self, other: &Self, modulus: &Self, out: &mut Self) {
        debug_assert!(
            !modulus.is_zero(),
            "modular subtraction requires a non-zero modulus"
        );
        let negative = self.cmp(other) == Ordering::Less;
        let underflowed = if negative {
            out.assign_difference(other, self)
        } else {
            out.assign_difference(self, other)
        };
        debug_assert!(!underflowed, "magnitude subtraction is ordered");
        if (*out).cmp(modulus) != Ordering::Less {
            out.sub_assign(modulus);
            if (*out).cmp(modulus) != Ordering::Less {
                let mut rem = Self::zero();
                let mut scratch = DivScratch::default();
                Division::rem_into(out, modulus, &mut rem, &mut scratch);
                if negative && !rem.is_zero() {
                    let complement_underflowed = out.assign_difference(modulus, &rem);
                    debug_assert!(!complement_underflowed, "remainder is below modulus");
                } else {
                    out.clone_from(&rem);
                }
                return;
            }
        }
        if negative && !out.is_zero() {
            out.negate_reduced_modulo(modulus);
        }
    }

    /// Computes `(self * other) % modulus`.
    ///
    /// A Montgomery domain reuses reciprocal state for repeated products
    /// with the same odd modulus.
    #[must_use]
    pub fn mul_mod(&self, other: &Self, modulus: &Self) -> Self {
        debug_assert!(
            !modulus.is_zero(),
            "modular multiplication requires a non-zero modulus"
        );
        if modulus.is_one() {
            return Self::zero();
        }
        let mut product = Self::zero();
        let mut div_scratch = DivScratch::default();
        let mut mul_scratch = MulScratch::default();
        product.assign_product_with_scratch(self, other, &mut mul_scratch);
        let mut remainder = Self::zero();
        Division::rem_into(&product, modulus, &mut remainder, &mut div_scratch);
        remainder
    }

    /// Computes `self^exp mod modulus` using sliding-window exponentiation.
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
        if exp.is_one() {
            return self.rem(modulus);
        }
        if self.is_zero() || self.is_one() {
            return self.clone();
        }

        // Small odd moduli use Montgomery reduction; wider or even moduli
        // use Barrett reduction at the configured crossover.
        if modulus.is_odd() && modulus.limbs().len() < MONTGOMERY_POW_MOD_THRESHOLD {
            if let [value] = modulus.limbs() {
                return LimbMontgomery::new(*value).pow(self, exp);
            }
            let domain = MontgomeryDomain::new::<true>(modulus);
            match modulus.limbs().len() {
                2 => return InlineMontgomery::<2>::pow(&domain, self, exp, false),
                3 => return InlineMontgomery::<3>::pow(&domain, self, exp, false),
                4 => return InlineMontgomery::<4>::pow(&domain, self, exp, false),
                _ => {}
            }
            return domain.pow(self, exp, &mut MontgomeryScratch::default(), false);
        }
        BarrettDomain::new(modulus).pow(self, exp, &mut MulScratch::default())
    }

    /// Performs Montgomery multiplication `(self * other * R^-1) mod modulus`.
    /// Operands at least as large as the modulus are reduced before REDC.
    #[must_use]
    pub fn montgomery_mul(&self, other: &Self, modulus: &Self) -> Self {
        debug_assert!(
            modulus.is_odd(),
            "Montgomery multiplication requires a non-zero odd modulus"
        );
        if self.is_zero() || other.is_zero() || modulus.is_one() {
            return Self::zero();
        }
        let domain = MontgomeryDomain::new::<false>(modulus);
        let mut div_scratch = DivScratch::default();
        let mut reduced_a = Self::zero();
        let mut reduced_b = Self::zero();
        let op_a = if self < modulus {
            self
        } else {
            Division::rem_into(self, modulus, &mut reduced_a, &mut div_scratch);
            &reduced_a
        };
        let op_b = if other < modulus {
            other
        } else {
            Division::rem_into(other, modulus, &mut reduced_b, &mut div_scratch);
            &reduced_b
        };
        let mut out = Self::zero();
        domain.mul_into_with_scratch(
            op_a,
            op_b,
            &mut out,
            &mut Self::zero(),
            &mut MontgomeryScratch::default(),
        );
        out
    }

    /// Performs modular reduction `self % modulus` using Barrett reduction.
    #[must_use]
    pub fn barrett_reduce(&self, modulus: &Self) -> Self {
        debug_assert!(
            !modulus.is_zero(),
            "Barrett reduction requires a non-zero modulus"
        );
        if modulus.is_one() {
            return Self::zero();
        }
        let input = self.limbs();
        let divisor = modulus.limbs();
        // With radix B, floor(x/m) < B iff floor(x/B) < m. At most
        // k input limbs guarantee this for a normalized k-limb modulus;
        // k+1 limbs need the high-prefix comparison. Such a quotient takes
        // linear division work and cannot amortize reciprocal construction.
        let single_limb_quotient = if input.len() <= divisor.len() {
            true
        } else if input.len().checked_sub(divisor.len()) == Some(1) {
            // SAFETY: len(input) = len(divisor)+1 >= 1, so removing the
            // initialized low limb leaves exactly len(divisor) limbs.
            let high = unsafe { input.get_unchecked(1..) };
            Self::cmp_limbs(high, divisor) == Ordering::Less
        } else {
            false
        };
        // Inputs wider than 2k exceed the Barrett reciprocal's range.
        if single_limb_quotient || input.len() > divisor.len().saturating_mul(2) {
            return self.rem(modulus);
        }
        let domain = BarrettDomain::new(modulus);
        let mut out = Self::zero();
        domain.reduce_into_with_barrett_scratch(
            self,
            &mut out,
            &mut MulScratch::default(),
            &mut BarrettScratch::default(),
        );
        out
    }

    /// Replaces a nonzero reduced residue with `modulus - self`.
    ///
    /// Zero extension aligns the widths. Each radix-B subtraction consumes
    /// its source limb before overwriting it, retaining the residue's buffer.
    pub fn negate_reduced_modulo(&mut self, modulus: &Self) {
        debug_assert!(
            !self.is_zero() && *self < *modulus,
            "residue is in (0, modulus)"
        );
        self.resize(modulus.limbs().len());
        let mut borrow = false;
        for (destination, &source) in self.limbs_mut().iter_mut().zip(modulus.limbs()) {
            let (difference, first_borrow) = source.overflowing_sub(*destination);
            let (result, second_borrow) = difference.overflowing_sub(usize::from(borrow));
            *destination = result;
            borrow = first_borrow || second_borrow;
        }
        debug_assert!(!borrow, "reduced residue is strictly below modulus");
        self.normalize();
    }
}
