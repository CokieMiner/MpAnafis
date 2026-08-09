//! Operator trait implementations for `MpUint` and `MpInt`.

use super::{InternalMpInt, MpInt, MpUint};
mod add;
mod bitwise;
mod div;
mod mul;
mod shift;
mod sub;

#[cfg(test)]
mod tests;
