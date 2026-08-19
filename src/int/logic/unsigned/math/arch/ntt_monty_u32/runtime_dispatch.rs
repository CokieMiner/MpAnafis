//! Runtime CPU feature dispatch for 31-bit Montgomery NTT kernels on x86-64.

use super::{
    NttMontyKernel, X86SimdTier, selected_x86_simd_tier, x86_64::ntt_monty_u32 as sse2_kernel,
    x86_64_avx2::ntt_monty_u32 as avx2_kernel,
};

#[inline]
pub fn selected_kernel() -> NttMontyKernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx2 => avx2_kernel,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}
