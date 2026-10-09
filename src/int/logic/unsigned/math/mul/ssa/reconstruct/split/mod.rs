//! Operand-to-coefficient splitting and fused pre-twisting.

use super::{LIMB_BITS, Limb, RingPeriods, SSA_PARALLEL_MIN_LIMB_WORK, SsaRing, SsaTransform};

mod parallel;
mod sequential;

pub use sequential::{SplitLayout, SsaCoefficients};

#[cfg(test)]
mod tests;
