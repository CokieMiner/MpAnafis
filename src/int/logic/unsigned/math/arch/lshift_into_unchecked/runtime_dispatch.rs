//! Runtime CPU feature dispatch for `lshift_into_unchecked` on `x86_64`.
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
    x86_64::lshift_into_unchecked as lshift_into_unchecked_sse2,
    x86_64_avx2::lshift_into_unchecked as lshift_into_unchecked_avx2,
};

type LshiftIntoKernel = unsafe fn(*mut Limb, *const Limb, usize, u32) -> Limb;

static KERNEL: OnceLock<LshiftIntoKernel> = OnceLock::new();

fn select_kernel() -> LshiftIntoKernel {
    match selected_x86_simd_tier() {
        X86SimdTier::Avx2 => lshift_into_unchecked_avx2,
        X86SimdTier::Sse2 => lshift_into_unchecked_sse2,
    }
}

#[inline]
pub fn selected_kernel() -> LshiftIntoKernel {
    *KERNEL.get_or_init(select_kernel)
}
