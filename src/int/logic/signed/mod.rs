//! Signed integer implementation — `InternalArbiInt`, arithmetic, bitwise, comparison.

use super::{INLINE_LIMBS, InternalArbiUint, Limb, UintRepr};

use assignment::negate_normalized_inplace;

mod arbiint;
mod arithmetic;
mod assignment;
mod bitwise;
mod cmp;
mod division;
mod theory;
mod wrapping;

pub use arbiint::InternalArbiInt;
