//! Integer arithmetic, modular residues, roots, and number theory.

use super::{DoubleLimb, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, ScratchBuffer, UintRepr};

mod add;
mod arch;
mod div;
mod gcd;
mod modular;
mod mul;
mod pow;
mod primes;
mod roots;
mod theory;
mod thresholds;
mod wrapping;

pub use add::Addition;
#[cfg(not(target_pointer_width = "16"))]
pub use arch::AddSubFromKernel;
pub use arch::ArchKernels;
pub use div::{DivScratch, Division};
pub use gcd::{Gcd, HgcdMatrix, HgcdWorkspace};
pub use modular::{
    BarrettDomain, BarrettScratch, LimbMontgomery, MontgomeryDomain, MontgomeryScratch,
};
pub use mul::{HighProduct, LowProduct, MulScratch, Multiplication, Schoolbook};
#[cfg(feature = "_internal-tune")]
pub use mul::{Karatsuba, MulPlan, SquarePlan, TierCeiling, Toom3, Toom4, Toom6, Toom8};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use mul::{Ssa, SsaMultiplicationPlan, SsaSquaringPlan, TransformChoice};
pub use pow::Exponentiation;
pub use primes::{ODD_COMPOSITE, Primality};
pub use roots::Roots;
pub use thresholds::*;

#[cfg(test)]
mod tests;
