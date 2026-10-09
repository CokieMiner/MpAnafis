//! Runtime CPU feature dispatch for `rshift_unchecked` on `x86_64`.
//!
//! The shared SIMD-tier selector resolves CPU features once; this module maps
//! that stable tier to the operation-specific function pointer.

use std::sync::OnceLock;

use super::{Limb, RshiftKernel, X86SimdTier, avx2_kernel, selected_x86_simd_tier, sse2_kernel};

static KERNEL: OnceLock<RshiftKernel> = OnceLock::new();

#[inline]
pub fn selected_kernel() -> RshiftKernel {
    *KERNEL.get_or_init(select_kernel)
}

fn select_kernel() -> RshiftKernel {
    match selected_x86_simd_tier() {
        // The in-place shift has no 512-bit backend.
        X86SimdTier::Avx512 | X86SimdTier::Avx2 => rshift_unchecked_simd,
        X86SimdTier::Sse2 => sse2_kernel,
    }
}

unsafe fn rshift_unchecked_simd(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    // Each four-limb vector requires one additional source limb.
    if len >= 5 {
        // SAFETY: selection of this wrapper proves AVX2, and the caller
        // establishes the writable span and shift bounds.
        unsafe { avx2_kernel(limbs, len, shift) }
    } else {
        // SAFETY: the baseline kernel uses x86-64 scalar instructions; the
        // caller establishes the initialized writable span and shift bounds.
        unsafe { sse2_kernel(limbs, len, shift) }
    }
}
