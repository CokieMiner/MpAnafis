//! GCD runners with retained operands and reusable Half-GCD scratch.

use core::fmt::{Debug, Formatter, Result as FmtResult};

use super::{Gcd, HgcdWorkspace, InternalMpUint, Limb, TuningResult};

/// Algorithms for multi-precision greatest common divisor computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GcdAlgorithm {
    /// Configured production dispatcher.
    Production,
    /// Lehmer reduction.
    Lehmer,
    /// Sub-quadratic recursive Half-GCD.
    HalfGcd,
}

/// Retained operands and scratch for GCD comparisons.
pub struct GcdRunner {
    left: InternalMpUint,
    right: InternalMpUint,
    workspace: HgcdWorkspace,
}

impl Debug for GcdRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("GcdRunner").finish_non_exhaustive()
    }
}

impl GcdRunner {
    /// Copies and normalizes the operand limb slices.
    ///
    /// Inputs are retained across runs. Half-GCD also reuses its workspace;
    /// output values, initial workspace growth, and other algorithms may allocate.
    #[must_use]
    pub fn new(left: &[Limb], right: &[Limb]) -> Self {
        Self {
            left: InternalMpUint::from_limbs(left.to_vec()),
            right: InternalMpUint::from_limbs(right.to_vec()),
            workspace: HgcdWorkspace::default(),
        }
    }

    /// Executes the specified GCD algorithm.
    pub fn run(&mut self, algorithm: GcdAlgorithm) -> impl AsRef<[Limb]> + Eq + Debug + use<> {
        TuningResult::new(match algorithm {
            GcdAlgorithm::Production => self.left.gcd(&self.right),
            GcdAlgorithm::Lehmer => Gcd::compute_lehmer(&self.left, &self.right),
            GcdAlgorithm::HalfGcd => {
                Gcd::compute_half_gcd(&self.left, &self.right, &mut self.workspace)
            }
        })
    }
}

/// Quotient simulation modes for Lehmer matrix formation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LehmerSimAlgorithm {
    /// Single-limb quotient simulation.
    Narrow,
    /// Double-limb quotient simulation.
    Wide,
}

/// Retained operands for single-limb and double-limb Lehmer simulation.
pub struct LehmerSimRunner {
    left: InternalMpUint,
    right: InternalMpUint,
}

impl Debug for LehmerSimRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("LehmerSimRunner")
            .finish_non_exhaustive()
    }
}

impl LehmerSimRunner {
    /// Constructs a simulation runner from operand limb slices.
    #[must_use]
    pub fn new(left: &[Limb], right: &[Limb]) -> Self {
        Self {
            left: InternalMpUint::from_limbs(left.to_vec()),
            right: InternalMpUint::from_limbs(right.to_vec()),
        }
    }

    /// Executes the Lehmer GCD with the specified simulation mode.
    pub fn run(
        &mut self,
        algorithm: LehmerSimAlgorithm,
    ) -> impl AsRef<[Limb]> + Eq + Debug + use<> {
        let force_wide = match algorithm {
            LehmerSimAlgorithm::Narrow => false,
            LehmerSimAlgorithm::Wide => true,
        };
        TuningResult::new(Gcd::compute_lehmer_configured::<false>(
            &self.left,
            &self.right,
            Some(force_wide),
        ))
    }
}

/// Vector update modes for applying the Lehmer transition matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LehmerUpdateAlgorithm {
    /// Fused equal-length pass updating both operands simultaneously.
    Fused,
    /// Independent sequential updates for each operand.
    Separate,
}

/// Retained operands for fused and separate Lehmer vector updates.
pub struct LehmerUpdateRunner {
    left: InternalMpUint,
    right: InternalMpUint,
}

impl Debug for LehmerUpdateRunner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("LehmerUpdateRunner")
            .finish_non_exhaustive()
    }
}

impl LehmerUpdateRunner {
    /// Constructs a vector update runner from operand limb slices.
    #[must_use]
    pub fn new(left: &[Limb], right: &[Limb]) -> Self {
        Self {
            left: InternalMpUint::from_limbs(left.to_vec()),
            right: InternalMpUint::from_limbs(right.to_vec()),
        }
    }

    /// Executes the Lehmer GCD with the specified vector update mode.
    pub fn run(
        &mut self,
        algorithm: LehmerUpdateAlgorithm,
    ) -> impl AsRef<[Limb]> + Eq + Debug + use<> {
        TuningResult::new(match algorithm {
            LehmerUpdateAlgorithm::Fused => {
                Gcd::compute_lehmer_configured::<true>(&self.left, &self.right, None)
            }
            LehmerUpdateAlgorithm::Separate => {
                Gcd::compute_lehmer_configured::<false>(&self.left, &self.right, None)
            }
        })
    }
}
