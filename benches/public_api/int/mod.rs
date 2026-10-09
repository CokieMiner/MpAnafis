//! Signed and unsigned public integer benchmarks with shared width ladders.
//! Each case specifies its operand signs, result contract, and comparison engine.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Benchmark expressions compare arbitrary-precision operators on validated domains; primitive sizing uses checked arithmetic"
)]

mod ladders;
mod support;

mod signed;
mod unsigned;
