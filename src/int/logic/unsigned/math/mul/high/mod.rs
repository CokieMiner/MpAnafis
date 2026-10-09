//! Exact high products and bounded recursive approximations.

use super::{
    Addition, ArchKernels, KARATSUBA_THRESHOLD, Limb, MulScratch, Multiplication, ScratchBuffer,
    Widths,
};

mod blocks;
mod entry;
mod trimmed;

pub use entry::HighProduct;

#[cfg(test)]
mod tests;
