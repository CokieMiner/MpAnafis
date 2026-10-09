//! Runtime dispatch for the x86-64 shared-source add/subtract kernel.

use std::sync::OnceLock;

use super::{
    AddSubFromKernel, X86Backend, adx_kernel, fallback_kernel, selected_x86_backend,
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
        X86Backend::Bmi2 | X86Backend::Baseline => fallback_kernel,
    }
}
