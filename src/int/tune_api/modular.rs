//! Modular exponentiation runners with retained operands and scratch.

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{
    BarrettDomain, InternalMpUint, Limb, MontgomeryDomain, MontgomeryScratch, MulScratch,
    TuningResult,
};

/// Algorithms for multi-precision modular exponentiation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModularPowAlgorithm {
    /// Configured production dispatcher.
    Production,
    /// Forced Montgomery reduction.
    Montgomery,
    /// Forced Barrett reduction.
    Barrett,
}

/// Retained operands and scratch for modular exponentiation comparisons.
pub struct ModularPowRunner {
    base: InternalMpUint,
    exp: InternalMpUint,
    modulus: InternalMpUint,
    scratch: MulScratch,
    montgomery_scratch: MontgomeryScratch,
}

/// Prepared raw Montgomery products, with domain setup outside the clock.
pub struct MontgomeryProductRunner {
    domain: MontgomeryDomain,
    left: InternalMpUint,
    right: InternalMpUint,
    output: InternalMpUint,
    product: InternalMpUint,
    scratch: MontgomeryScratch,
}

impl Debug for MontgomeryProductRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("MontgomeryProductRunner")
            .finish_non_exhaustive()
    }
}

impl MontgomeryProductRunner {
    /// Prepares `a*b*R^-1 mod M`, where `R=B^n` and `n=M.len()`.
    ///
    /// # Panics
    ///
    /// Panics unless `M` is positive and odd and both operands are below `M`.
    #[must_use]
    pub fn new(left_limbs: &[Limb], right_limbs: &[Limb], modulus_limbs: &[Limb]) -> Self {
        let left = InternalMpUint::from_limbs(left_limbs.to_vec());
        let right = InternalMpUint::from_limbs(right_limbs.to_vec());
        let modulus = InternalMpUint::from_limbs(modulus_limbs.to_vec());
        assert!(
            modulus.is_odd() && left < modulus && right < modulus,
            "Montgomery product requires an odd modulus and reduced operands"
        );
        Self {
            domain: MontgomeryDomain::new::<false>(&modulus),
            left,
            right,
            output: InternalMpUint::zero(),
            product: InternalMpUint::zero(),
            scratch: MontgomeryScratch::default(),
        }
    }

    /// Runs the prepared production multiplication with reusable scratch.
    pub fn run(&mut self) -> &[Limb] {
        self.domain.mul_into_with_scratch(
            &self.left,
            &self.right,
            &mut self.output,
            &mut self.product,
            &mut self.scratch,
        );
        self.output.limbs()
    }
}

impl Debug for ModularPowRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("ModularPowRunner")
            .finish_non_exhaustive()
    }
}

impl ModularPowRunner {
    /// Copies and normalizes the base, exponent, and odd modulus limb slices.
    ///
    /// Invocations of [`Self::run`] reuse the operands and multiplication scratch.
    /// Each invocation constructs its reduction domain, power table, and result,
    /// including their allocations.
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is zero or not odd.
    #[must_use]
    pub fn new(base: &[Limb], exp: &[Limb], modulus: &[Limb]) -> Self {
        assert!(
            modulus.iter().any(|&limb| limb != 0)
                && modulus.first().is_some_and(|&limb| (limb & 1) != 0),
            "modular exponentiation runner requires a nonzero odd modulus"
        );
        let base_uint = InternalMpUint::from_limbs(base.to_vec());
        let exp_uint = InternalMpUint::from_limbs(exp.to_vec());
        let mod_uint = InternalMpUint::from_limbs(modulus.to_vec());
        Self {
            base: base_uint,
            exp: exp_uint,
            modulus: mod_uint,
            scratch: MulScratch::default(),
            montgomery_scratch: MontgomeryScratch::default(),
        }
    }

    /// Executes the specified modular exponentiation algorithm.
    pub fn run(
        &mut self,
        algorithm: ModularPowAlgorithm,
    ) -> impl AsRef<[Limb]> + Eq + Debug + use<> {
        // Modulo one, every power has the canonical residue zero, including
        // the exponent-zero identity. No reduction domain is needed.
        if self.modulus.is_one() {
            return TuningResult::new(InternalMpUint::zero());
        }
        TuningResult::new(match algorithm {
            ModularPowAlgorithm::Production => self.base.pow_mod(&self.exp, &self.modulus),
            ModularPowAlgorithm::Montgomery => MontgomeryDomain::new::<true>(&self.modulus).pow(
                &self.base,
                &self.exp,
                &mut self.montgomery_scratch,
                false,
            ),
            ModularPowAlgorithm::Barrett => {
                BarrettDomain::new(&self.modulus).pow(&self.base, &self.exp, &mut self.scratch)
            }
        })
    }
}
