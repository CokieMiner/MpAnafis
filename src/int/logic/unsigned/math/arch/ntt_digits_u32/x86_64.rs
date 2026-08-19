//! SSE2-baseline provider for 16-bit NTT digit packing.

use super::{NttDigitsKernels, limbs_to_digits_16_scalar};

pub fn ntt_digits_u32() -> NttDigitsKernels {
    NttDigitsKernels {
        pack_16: limbs_to_digits_16_scalar,
    }
}
