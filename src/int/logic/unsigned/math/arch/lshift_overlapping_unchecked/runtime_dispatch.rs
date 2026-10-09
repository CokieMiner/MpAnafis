//! Runtime dispatch for the overlap-safe left-shift kernel.

use std::sync::OnceLock;

use super::{
    Limb, LshiftOverlappingKernel, X86SimdTier, avx2_kernel, avx512_kernel,
    selected_x86_simd_tier, sse2_kernel,
};

static KERNEL: OnceLock<LshiftOverlappingKernel> = OnceLock::new();

fn select_kernel() -> LshiftOverlappingKernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx512 => avx512_tier_kernel,
        X86SimdTier::Avx2 => avx2_tier_kernel,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}

#[inline]
pub fn selected_kernel() -> LshiftOverlappingKernel {
    *KERNEL.get_or_init(select_kernel)
}

unsafe fn avx512_tier_kernel(
    limbs: *mut Limb,
    len: usize,
    offset: usize,
    shift: u32,
) -> Limb {
    if len >= 8 {
        // SAFETY: selection of this wrapper proves AVX-512F, and the caller
        // establishes the complete span and shift bounds.
        unsafe { avx512_kernel(limbs, len, offset, shift) }
    } else {
        // SAFETY: every AVX-512F host admitted by the selector also passed the
        // AVX2 feature check; the caller establishes the remaining contract.
        unsafe { avx2_tier_kernel(limbs, len, offset, shift) }
    }
}

unsafe fn avx2_tier_kernel(
    limbs: *mut Limb,
    len: usize,
    offset: usize,
    shift: u32,
) -> Limb {
    if len >= 5 {
        // SAFETY: selection of this wrapper proves AVX2, and the caller
        // establishes the complete span and shift bounds.
        unsafe { avx2_kernel(limbs, len, offset, shift) }
    } else {
        // SAFETY: SSE2 is mandatory on x86-64 and the caller establishes the
        // complete span and shift bounds.
        unsafe { sse2_kernel(limbs, len, offset, shift) }
    }
}
