//! Property-based integration tests for `MpUint` and `MpInt`.
//!
//! Each module owns one behavior. Shared public constructors and strategies
//! generate inputs without accessing private integer storage.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    reason = "Test files: arithmetic operators are the API under test; casts use property-bounded values"
)]

mod arithmetic;
mod bitwise;
mod bounded;
mod bytes;
mod comparison;
mod conversions;
mod division;
#[cfg(feature = "rayon")]
mod executor;
mod fused;
mod gcd;
mod memory;
mod modular;
mod mul;
#[cfg(feature = "num-traits")]
mod num_traits;
mod ops;
mod powers;
mod precision;
mod primality;
mod roots;
mod strategies;
mod stress;
mod string;
mod support;
mod theory;
mod traits;
mod types;
