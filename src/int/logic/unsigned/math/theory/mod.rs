//! Number-theoretic operations for [`InternalArbiUint`].

use super::{
    ArchKernels, BarrettDomain, BarrettScratch, DivScratch, Division, Gcd, InternalArbiUint, Limb,
    MontgomeryDomain, MulScratch,
};

#[cfg(test)]
use self::totient::compute_abs_diff;

mod factorial;
mod jacobi;
mod totient;

#[cfg(test)]
mod tests;
