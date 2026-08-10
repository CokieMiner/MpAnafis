//! Integer domain benchmarks: `ArbiUint` in [`unsigned`] and `ArbiInt` in
//! [`signed`], both against their Rug/GMP counterparts.
//!
//! `ArbiInt` delegates its magnitude arithmetic to `ArbiUint` and adds sign
//! handling, so the two trees are deliberately parallel and draw from the same
//! [`ladders`]: a cell that is slower on the signed side than on the unsigned
//! side is measuring the sign layer and nothing else.

pub mod ladders;
pub mod support;

mod signed;
mod unsigned;
