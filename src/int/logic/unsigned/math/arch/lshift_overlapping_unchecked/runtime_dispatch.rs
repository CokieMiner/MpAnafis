//! Runtime dispatch for the overlap-safe left-shift kernel.

use std::sync::OnceLock;

use super::{
    Limb, X86SimdTier, selected_x86_simd_tier,
    x86_64::lshift_overlapping_unchecked as sse2_kernel,
    x86_64_avx2::lshift_overlapping_unchecked as avx2_kernel,
    x86_64_avx512::lshift_overlapping_unchecked as avx512_kernel,
};

type Kernel = unsafe fn(*mut Limb, usize, usize, u32) -> Limb;

static KERNEL: OnceLock<Kernel> = OnceLock::new();

fn select_kernel() -> Kernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx512 => avx512_kernel,
        X86SimdTier::Avx2 => avx2_kernel,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}

#[inline]
pub fn selected_kernel() -> Kernel {
    *KERNEL.get_or_init(select_kernel)
}
