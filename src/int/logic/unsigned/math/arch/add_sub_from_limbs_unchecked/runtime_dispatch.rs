//! Runtime dispatch for the x86-64 shared-source add/subtract kernel.

use std::sync::OnceLock;

use super::{
    AddSubFromKernel, X86Backend, fallback::add_sub_from_limbs_unchecked as fallback_kernel,
    selected_x86_backend, x86_64_adx::add_sub_from_limbs_unchecked as adx_kernel,
};

static KERNEL: OnceLock<AddSubFromKernel> = OnceLock::new();

fn select_kernel() -> AddSubFromKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Adx => adx_kernel,
        X86Backend::Bmi2 | X86Backend::Baseline => fallback_kernel,
    }
}

/// Return the selected shared-input addition/subtraction kernel.
#[inline]
pub fn selected_kernel() -> AddSubFromKernel {
    *KERNEL.get_or_init(select_kernel)
}
