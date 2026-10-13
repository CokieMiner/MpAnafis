//! Extended-GCD cofactors and modular inverse reconstruction.

use super::{
    ArchKernels, DivScratch, Division, EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS,
    EXTENDED_GCD_COFACTOR_BATCH_RATIO, EXTENDED_GCD_WIDE_THRESHOLD,
    EXTENDED_HGCD_CROSSOVER_THRESHOLD, Gcd, HgcdMatrix, HgcdWorkspace, InternalMpUint, Limb,
};

mod bezout;
mod euclid;

#[cfg(test)]
mod tests;
