//! GCD arithmetic, simulation, matrix action, storage, and crossover tests.

use super::{
    ArchKernels, DivScratch, DoubleLimb, Gcd, HGCD_BLOCK_THRESHOLD, HGCD_CROSSOVER_THRESHOLD,
    HgcdFrame, HgcdMatrix, HgcdWorkspace, InternalMpUint, LEHMER_FUSED_UPDATE_MAX_LIMBS, LIMB_BITS,
    Limb, SignedLimbCarry, WIDE_LEHMER_THRESHOLD,
};

mod boundaries;
mod carry;
mod hgcd2;
mod jacobi;
mod properties;
mod scalar;
mod simulation;
mod transitions;
