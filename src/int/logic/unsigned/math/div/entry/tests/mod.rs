//! Division dispatch, exactness, quotient certificates, and reusable outputs.

use crate::int::logic::unsigned::math::div::{
    BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_THRESHOLD,
    DIVISION_STACK_LIMBS, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb,
    NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_THRESHOLD, PreparedDivisor, QuotientAndRemainder,
    QuotientOnly, RemainderCertificate, ScratchBuffer, TruncatedPrefix,
};

mod dispatch;
mod divisibility;
mod exact;
mod guarded;
mod identities;
mod normalized;
mod quotient;
mod reusable;
