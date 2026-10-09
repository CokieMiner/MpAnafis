//! Number-theoretic operations for [`InternalMpUint`].

use super::{
    ArchKernels, BarrettDomain, BarrettScratch, DivScratch, Division, DoubleLimb, Gcd,
    INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, LimbMontgomery, MulScratch, Multiplication,
    ODD_COMPOSITE, Primality, Roots, Schoolbook,
};

mod factorial;
mod totient;
mod trial;

pub use trial::{PRIMALITY_PREFIX, PRIME_COUNT, TRIAL_BOUND, Totient};

#[cfg(test)]
mod tests;
