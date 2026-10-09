//! Modular arithmetic, reduction domains, and exponentiation.

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, ArchKernels, DivScratch, Division, DoubleLimb, Exponentiation, HighProduct,
    INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, LowProduct, MONTGOMERY_CIOS_MAX_LIMBS,
    MONTGOMERY_POW_MOD_THRESHOLD, MulScratch, Schoolbook, ScratchBuffer,
};

mod barrett;
mod inline;
mod inverse;
mod montgomery;
mod operations;
mod power;
mod reduction;

pub use barrett::{BarrettDomain, BarrettScratch};
pub use inline::InlineMontgomery;
pub use montgomery::{LimbMontgomery, MontgomeryDomain};
pub use reduction::MontgomeryScratch;

#[cfg(test)]
mod tests;
