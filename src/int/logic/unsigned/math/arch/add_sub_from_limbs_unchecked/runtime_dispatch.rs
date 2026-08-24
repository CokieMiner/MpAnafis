//! Runtime dispatch for the x86-64 shared-source add/subtract kernel.

use std::sync::OnceLock;

use super::{
    AddSubFromKernel, X86Backend, X86SimdTier,
    fallback::add_sub_from_limbs_unchecked as fallback_kernel, selected_x86_backend,
    selected_x86_simd_tier, x86_64_adx::add_sub_from_limbs_unchecked as adx_kernel,
    x86_64_avx2::add_sub_from_limbs_unchecked as avx2_kernel,
};

static KERNEL: OnceLock<AddSubFromKernel> = OnceLock::new();

/// Return the selected shared-input addition/subtraction kernel.
#[inline]
pub fn selected_kernel() -> AddSubFromKernel {
    *KERNEL.get_or_init(select_kernel)
}

fn select_kernel() -> AddSubFromKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Adx => adx_kernel,
        X86Backend::Bmi2 | X86Backend::Baseline => match selected_x86_simd_tier() {
            // The AVX-512 tier has no 512-bit add/sub backend (the software
            // carry chains lose to the scalar kernels), so an AVX-512 host
            // runs the 256-bit one; its vector width is not why it was chosen.
            X86SimdTier::Avx512 | X86SimdTier::Avx2 => avx2_kernel,
            X86SimdTier::Sse2 => fallback_kernel,
        },
    }
}
