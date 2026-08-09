//! `MpInt` benchmarks.
//!
//! `MpInt` holds an `MpUint` magnitude and a sign, and delegates every limb
//! algorithm to it. These modules therefore exist to measure the *delegation*,
//! not the arithmetic: each one runs the same operation the corresponding
//! [`unsigned`](super::unsigned) module runs, on the same widths, so the
//! difference between the two trees is the sign layer.
//!
//! Two places need signed operands specifically rather than as a matter of
//! symmetry, and both get negative inputs here:
//!
//! - [`sign`], which has no unsigned counterpart at all.
//! - [`division`], where truncation, flooring, ceiling and Euclidean rounding
//!   give four different answers once the dividend is negative. On unsigned
//!   operands all four agree, so that is the only place the rounding wrappers
//!   have anything to do.

mod arithmetic;
mod bitwise;
mod comparison;
mod conversion;
mod division;
mod sign;
mod theory;
