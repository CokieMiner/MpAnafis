//! Runtime CPU feature dispatch for `rshift_unchecked` on `x86_64`.
//!
//! The shared SIMD-tier selector resolves CPU features once; this module maps
//! that stable tier to the operation-specific function pointer.
//!
//! ## Testing override (debug builds only)
//!
//! ```bash
//! MP_ANAFIS_TEST_BACKEND=avx2 cargo test test_
//! MP_ANAFIS_TEST_BACKEND=sse2 cargo test test_
//! ```

use std::sync::OnceLock;

use super::{
    Limb, X86SimdTier, selected_x86_simd_tier,
    x86_64::rshift_unchecked as rshift_unchecked_sse2,
    x86_64_avx2::rshift_unchecked as rshift_unchecked_avx2,
};

type RshiftKernel = unsafe fn(*mut Limb, usize, u32) -> Limb;

static KERNEL: OnceLock<RshiftKernel> = OnceLock::new();

fn select_kernel() -> RshiftKernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx2 => rshift_unchecked_avx2,
        X86SimdTier::Sse2 => rshift_unchecked_sse2,
    }
}

#[inline]
pub fn selected_kernel() -> RshiftKernel {
    *KERNEL.get_or_init(select_kernel)
}
