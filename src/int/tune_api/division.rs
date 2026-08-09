//! Reusable division candidates for crossover measurements.

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{DivScratch, Division, InternalMpUint, Limb};

/// Algorithms for multi-precision division.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Configured production dispatcher.
    Production,
    /// Knuth's Algorithm D basecase.
    AlgorithmD,
    /// Burnikel-Ziegler division.
    BurnikelZiegler,
    /// Newton-Raphson division.
    NewtonRaphson,
}

/// Pre-allocated state for comparing division algorithms.
pub struct Tuner {
    num: InternalMpUint,
    den: InternalMpUint,
    q: InternalMpUint,
    r: InternalMpUint,
    scratch: DivScratch,
}

impl Debug for Tuner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Tuner").finish_non_exhaustive()
    }
}

impl Tuner {
    /// Create a benchmark pair from raw limbs, allocating only here.
    ///
    /// # Panics
    ///
    /// Panics if `den` represents zero.
    #[must_use]
    pub fn new(num: &[Limb], den: &[Limb]) -> Self {
        assert!(
            den.iter().any(|&limb| limb != 0),
            "division tuner denominator is zero"
        );
        let numerator = InternalMpUint::from_limbs(num.to_vec());
        let denominator = InternalMpUint::from_limbs(den.to_vec());
        assert!(
            numerator >= denominator,
            "division tuner numerator is smaller than its denominator"
        );
        Self {
            num: numerator,
            den: denominator,
            q: InternalMpUint::zero(),
            r: InternalMpUint::zero(),
            scratch: DivScratch::default(),
        }
    }

    /// Executes the specified division algorithm.
    pub fn run(&mut self, algorithm: Algorithm) {
        match algorithm {
            Algorithm::Production => Division::div_rem_into(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
            Algorithm::AlgorithmD => Division::algorithm_d(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
            Algorithm::BurnikelZiegler => Division::burnikel_ziegler(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
            Algorithm::NewtonRaphson => Division::newton(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
        }
    }

    /// The quotient of the last [`Self::run`], as raw limbs.
    ///
    /// Exists so crossover measurements can verify that two candidate
    /// algorithms produced identical quotients before comparing their times.
    #[must_use]
    pub fn quotient_limbs(&self) -> &[Limb] {
        self.q.limbs()
    }

    /// The remainder of the last [`Self::run`], as raw limbs.
    ///
    /// Exists so crossover measurements can verify that two candidate
    /// algorithms produced identical remainders before comparing their times.
    #[must_use]
    pub fn remainder_limbs(&self) -> &[Limb] {
        self.r.limbs()
    }
}
