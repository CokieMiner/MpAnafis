//! Runtime dispatch for x86-64 BMI2 shifted-high subtraction.

use std::sync::OnceLock;

use super::{
    SubShiftedHighKernel, X86Backend, bmi2_kernel, fallback_kernel, selected_x86_backend,
};

static KERNEL: OnceLock<SubShiftedHighKernel> = OnceLock::new();

/// Returns the shifted-source subtraction backend selected for this CPU.
#[inline]
pub fn selected_kernel() -> SubShiftedHighKernel {
    *KERNEL.get_or_init(select_kernel)
}

fn select_kernel() -> SubShiftedHighKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Bmi2 => bmi2_kernel,
        X86Backend::Adx | X86Backend::Baseline => fallback_kernel,
    }
}
