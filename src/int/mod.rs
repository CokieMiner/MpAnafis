//! Integer types — re-exports the full public API.

use logic::InternalPrecisionContext;
#[cfg(feature = "_internal-tune")]
use logic::{
    ArchKernels, DivScratch, Division, FormatCache, Karatsuba, Lopsided, LowProduct, MulScratch,
    Multiplication, Ntt, Schoolbook, ScratchBuffer, Ssa, Toom3, Toom4, Toom6, Toom8, Toom32,
    Toom43, TransformBench, TransformChoice,
};
use types::{DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES};

mod api;
mod logic;
#[cfg(feature = "_internal-tune")]
#[doc(hidden)]
pub mod tune_api;
mod types;

#[cfg(test)]
mod tests;

pub use api::*;
pub use logic::{InternalArbiInt, InternalArbiUint};
#[cfg(not(feature = "_internal-tune"))]
pub use types::Limb;
#[cfg(feature = "_internal-tune")]
#[doc(hidden)]
pub use types::Limb;
