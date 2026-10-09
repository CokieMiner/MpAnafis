//! Differential integer fuzz cases shared by libFuzzer and deterministic tests.

mod bit_reference;
mod input;
mod policy;
mod reference;
pub mod signed;
mod theory_reference;
pub mod unsigned;

pub use bit_reference::BitReference;
pub use input::Input;
pub use policy::{Bounds, PolicyResults};
pub use reference::{assert_integer, assert_optional, float32, float64, modular, signed_bytes};
pub use theory_reference::TheoryReference;

#[cfg(test)]
mod tests;
