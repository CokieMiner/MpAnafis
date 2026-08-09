//! Tests for the division tower.
//!
//! Organised by what each file pins down rather than by which source file the
//! code lives in:
//!
//! - [`dispatch`]: the entry points agree with each other, whichever crossover
//!   the operands land on.
//! - [`quotient`]: the truncated quotient path agrees with the full engine.
//! - [`newton`]: reciprocal division agrees with Algorithm D.

mod dispatch;
mod newton;
mod quotient;

use super::{BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb};
