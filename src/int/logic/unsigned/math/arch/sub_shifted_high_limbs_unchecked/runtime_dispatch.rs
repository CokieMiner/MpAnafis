//! Runtime dispatch for x86-64 BMI2 shifted-high subtraction.

use std::sync::OnceLock;

use super::{
    SubShiftedHighKernel, X86Backend,
    fallback::sub_shifted_high_limbs_unchecked as fallback_kernel, selected_x86_backend,
    x86_64_bmi2::sub_shifted_high_limbs_unchecked as bmi2_kernel,
};

static KERNEL: OnceLock<SubShiftedHighKernel> = OnceLock::new();

fn select_kernel() -> SubShiftedHighKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Bmi2 => bmi2_kernel,
        X86Backend::Adx | X86Backend::Baseline => fallback_kernel,
    }
}

/// Return the selected shifted-high subtraction kernel.
#[inline]
pub fn selected_kernel() -> SubShiftedHighKernel {
    *KERNEL.get_or_init(select_kernel)
}
