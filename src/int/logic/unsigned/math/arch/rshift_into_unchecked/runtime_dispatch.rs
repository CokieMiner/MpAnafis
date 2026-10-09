//! Runtime CPU feature dispatch for `rshift_into_unchecked` on `x86_64`.
//!
//! The shared SIMD-tier selector resolves CPU features once; this module maps
//! that stable tier to the operation-specific function pointer.
//!
use std::sync::OnceLock;

use super::{
    RshiftIntoKernel, X86SimdTier, avx2_kernel, avx512_kernel, selected_x86_simd_tier, sse2_kernel,
};

static KERNEL: OnceLock<RshiftIntoKernel> = OnceLock::new();

fn select_kernel() -> RshiftIntoKernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx512 => avx512_kernel,
        X86SimdTier::Avx2 => avx2_kernel,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}

#[inline]
pub fn selected_kernel() -> RshiftIntoKernel {
    *KERNEL.get_or_init(select_kernel)
}
