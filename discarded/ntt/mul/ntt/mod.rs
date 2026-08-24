//! 50-bit floating-point Harvey NTT multiplication module.

use super::{ArchKernels, LIMB_BITS, Limb};

mod conversion;
mod crt;
mod plan;
mod prime;
mod transform;

#[cfg(test)]
mod tests;

pub use plan::Ntt;
pub use prime::{
    FLOAT_PINV_1, FLOAT_PINV_2, FLOAT_PINV_3, FLOAT_PRIME_1, FLOAT_PRIME_1_INT, FLOAT_PRIME_2,
    FLOAT_PRIME_2_INT, FLOAT_PRIME_3, FLOAT_PRIME_3_INT, FLOAT_ROOT_1, FLOAT_ROOT_2, FLOAT_ROOT_3,
};
