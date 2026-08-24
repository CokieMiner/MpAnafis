//! Number-theoretic operations for [`InternalMpUint`].

use super::{
    ArchKernels, BarrettDomain, BarrettScratch, DivScratch, Division, Gcd, InternalMpUint, Limb,
    MontgomeryDomain, MulScratch,
};

#[cfg(test)]
use self::totient::compute_abs_diff;

mod factorial;
mod jacobi;
mod totient;

#[cfg(test)]
mod tests;
