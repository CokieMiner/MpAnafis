//! Primality testing for [`InternalMpUint`].

use super::{
    ArchKernels, Gcd, InternalMpUint, LIMB_BITS, Limb, LimbMontgomery, MontgomeryDomain,
    MontgomeryScratch,
};

mod baillie_psw;
mod miller_rabin;
mod native;
mod operations;
mod search;
mod sieve;
mod tables;

pub use operations::Primality;
pub use sieve::ODD_COMPOSITE;
pub use tables::{TRIAL_PRIME_LIMIT, TRIAL_SCREEN};

#[cfg(test)]
mod tests;
