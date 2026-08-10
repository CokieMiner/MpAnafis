//! Operator trait implementations for `ArbiUint` and `ArbiInt`.

use super::{ArbiInt, ArbiUint, InternalArbiInt};
mod add;
mod bitwise;
mod div;
mod mul;
mod shift;
mod sub;

#[cfg(test)]
mod tests;
