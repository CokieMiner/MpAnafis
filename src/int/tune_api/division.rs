//! Division runners with retained operands and reusable output and scratch buffers.

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{DivScratch, Division, InternalMpUint, Limb};

/// Algorithms for multi-precision division.
///
/// The const parameter selects whether the caller consumes the remainder.
/// Production, Algorithm D, and Newton omit their final remainder publication
/// for quotient-only runs. Burnikel-Ziegler retains its complete remainder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DivisionAlgorithm<const WRITE_REMAINDER: bool = true> {
    /// Configured production dispatcher.
    Production,
    /// Knuth's Algorithm D basecase.
    AlgorithmD,
    /// Burnikel-Ziegler division.
    BurnikelZiegler,
    /// Newton-Raphson division.
    NewtonRaphson,
}

/// Retained operands, output buffers, and scratch for division comparisons.
pub struct DivisionRunner {
    num: InternalMpUint,
    den: InternalMpUint,
    q: InternalMpUint,
    r: InternalMpUint,
    scratch: DivScratch,
}

impl Debug for DivisionRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("DivisionRunner")
            .finish_non_exhaustive()
    }
}

impl DivisionRunner {
    /// Copies and normalizes numerator and denominator limb slices.
    ///
    /// Invocations of [`Self::run`] reuse output and scratch buffers. Their first
    /// use, a different algorithm, or an algorithm-specific temporary may allocate.
    ///
    /// # Panics
    ///
    /// Panics if `den` represents zero or `num < den`.
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

    /// Executes the specified division algorithm under the algorithm's remainder policy.
    ///
    /// A quotient-only run leaves the remainder accessor unspecified. The
    /// production dispatcher may select a quotient-specific algorithm.
    pub fn run<const WRITE_REMAINDER: bool>(
        &mut self,
        algorithm: DivisionAlgorithm<WRITE_REMAINDER>,
    ) {
        match algorithm {
            DivisionAlgorithm::Production => {
                if WRITE_REMAINDER {
                    Division::div_rem_into(
                        &self.num,
                        &self.den,
                        &mut self.q,
                        &mut self.r,
                        &mut self.scratch,
                    );
                } else {
                    Division::div_into::<true, false>(
                        &self.num,
                        &self.den,
                        &mut self.q,
                        &mut self.scratch,
                    );
                }
            }
            DivisionAlgorithm::AlgorithmD => {
                let _ = Division::algorithm_d::<true, WRITE_REMAINDER, false, false>(
                    self.num.limbs(),
                    self.den.limbs(),
                    &mut self.q,
                    &mut self.r,
                    &mut self.scratch,
                );
            }
            DivisionAlgorithm::BurnikelZiegler => Division::burnikel_ziegler::<true>(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
            DivisionAlgorithm::NewtonRaphson => Division::newton::<true, WRITE_REMAINDER, false>(
                &self.num,
                &self.den,
                &mut self.q,
                &mut self.r,
                &mut self.scratch,
            ),
        }
    }

    /// Returns the canonical quotient limbs from the last [`Self::run`].
    #[must_use]
    pub fn quotient_limbs(&self) -> &[Limb] {
        self.q.limbs()
    }

    /// Returns the remainder limbs from the last remainder-producing run.
    ///
    /// The result is defined after a run with `WRITE_REMAINDER = true`.
    /// Quotient-only runs may leave or overwrite this buffer.
    #[must_use]
    pub fn remainder_limbs(&self) -> &[Limb] {
        self.r.limbs()
    }
}
