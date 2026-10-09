//! Ring arithmetic, normalization, fixed factors, and carry boundaries.

use super::{LIMB_BITS, Limb, SsaRing};

mod addition;
mod factors;
mod negation;
mod normalization;
mod oracle;
mod scaling;
mod shifts;
mod wide_shifts;

pub use oracle::{oracle_half, oracle_negation};
