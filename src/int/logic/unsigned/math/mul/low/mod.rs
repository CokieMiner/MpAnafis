//! Low products modulo a limb radix power and the Mulders split schedule.

use super::{
    Addition, ArchKernels, LOW_PRODUCT_FULL_THRESHOLD, LOW_PRODUCT_RECURSIVE_THRESHOLD, Limb,
    LimbOutput, MulScratch, Multiplication, Schoolbook, ScratchBuffer, TierCeiling,
};

mod entry;

pub use entry::LowProduct;

#[cfg(test)]
mod tests;
