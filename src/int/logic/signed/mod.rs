//! Sign-magnitude arithmetic and two's-complement operations.

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

mod arithmetic;
mod assignment;
mod bitwise;
mod cmp;
mod division;
mod mpint;
mod theory;
mod wrapping;

pub use mpint::InternalMpInt;

#[cfg(test)]
mod tests;
