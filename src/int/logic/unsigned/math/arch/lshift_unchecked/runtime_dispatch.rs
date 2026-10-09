//! Runtime CPU feature dispatch for `lshift_unchecked` on `x86_64`.
//!
//! The shared SIMD-tier selector resolves CPU features once; this module maps
//! that stable tier to the operation-specific function pointer.
//!
use std::sync::OnceLock;

use super::{
    Limb, LshiftKernel, X86SimdTier, avx2_kernel, selected_x86_simd_tier, sse2_kernel,
};

static KERNEL: OnceLock<LshiftKernel> = OnceLock::new();

fn select_kernel() -> LshiftKernel {
    match selected_x86_simd_tier() {
        // The in-place shift has no 512-bit backend; an AVX-512 host runs the
        // 256-bit one once there are enough limbs to execute its vector loop.
        X86SimdTier::Avx512 | X86SimdTier::Avx2 => lshift_unchecked_simd,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}

#[inline]
pub fn selected_kernel() -> LshiftKernel {
    *KERNEL.get_or_init(select_kernel)
}

unsafe fn lshift_unchecked_simd(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    // The AVX2 kernel cannot enter its four-limb loop below five limbs.  Keep
    // those common inline values on the baseline `shld` kernel instead of its
    // scalar tail.
    if len >= 5 {
        // SAFETY: selection of this wrapper proves AVX2, and the caller
        // establishes the writable span and shift bounds.
        unsafe { avx2_kernel(limbs, len, shift) }
    } else {
        // SAFETY: SSE2 is mandatory on x86-64 and the caller establishes the
        // writable span and shift bounds.
        unsafe { sse2_kernel(limbs, len, shift) }
    }
}
