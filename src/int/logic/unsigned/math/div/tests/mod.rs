//! Division identities, quotient policies, reciprocal bounds, and reusable storage.

use super::{
    BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_THRESHOLD,
    DIVISION_SMALL_QUOTIENT_MAX, DIVISION_STACK_LIMBS, DivScratch, Division, DoubleLimb,
    EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS, EXTENDED_HGCD_CROSSOVER_THRESHOLD, Gcd, HgcdMatrix,
    HgcdWorkspace, InternalMpUint, LIMB_BITS, Limb, NEWTON_QUOTIENT_THRESHOLD,
    NEWTON_RAPHSON_THRESHOLD, PreparedDivisor, ScratchBuffer,
};

mod approximate;
mod blocks;
mod dispatch;
mod divisibility;
mod exact;
mod extended;
mod identities;
mod limbs;
mod normalized;
mod prepared;
mod quotient;
mod reusable;
mod short;
mod small;
