//! Math operations — add, sub, mul, div, pow, wrapping, shift, modular.

use super::{DoubleLimb, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, ScratchBuffer, UintRepr};

use add::Addition;
use gcd::Gcd;
use montgomery::MontgomeryDomain;
#[cfg(not(feature = "_internal-tune"))]
use mul::{LowProduct, Multiplication};

mod add;
pub mod arch;
mod barrett;
pub mod div;
mod gcd;
mod modular;
mod montgomery;
pub mod mul;
mod pow;
mod primes;
mod roots;
mod theory;
/// Algorithmic crossover thresholds for integer operations.
mod thresholds;
mod wrapping;

#[cfg(test)]
mod tests;

pub use arch::ArchKernels;
pub use barrett::{BarrettDomain, BarrettScratch};
pub use div::{DivScratch, Division};
pub use mul::MulScratch;
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use mul::TransformBench;
#[cfg(feature = "_internal-tune")]
pub use mul::{
    Karatsuba, Lopsided, LowProduct, Multiplication, Ntt, Schoolbook, Toom3, Toom4, Toom6, Toom8,
    Toom32, Toom43,
};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use mul::{Ssa, TransformChoice};
pub use thresholds::*;
