//! Integer domain benchmarks: `MpUint` in [`unsigned`] and `MpInt` in
//! [`signed`], both against their Rug/GMP counterparts.
//!
//! `MpInt` delegates its magnitude arithmetic to `MpUint` and adds sign
//! handling, so the two trees are deliberately parallel and draw from the same
//! [`ladders`]: a cell that is slower on the signed side than on the unsigned
//! side is measuring the sign layer and nothing else.

pub mod ladders;
pub mod support;

mod signed;
mod unsigned;
