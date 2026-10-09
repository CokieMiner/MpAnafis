//! Runtime CPU-feature dispatch for `monty_redc_step_unchecked` on `x86_64`.
//!
//! The shared architecture selector resolves CPU features once. Baseline x86-64
//! uses the portable dual-carry kernel; BMI2 and ADX+BMI2 select their dedicated
//! assembly kernels.

use std::sync::OnceLock;

use super::{
    MontyKernel, X86Backend, adx_kernel, bmi2_kernel, fallback_kernel, selected_x86_backend,
};

static KERNEL: OnceLock<MontyKernel> = OnceLock::new();

fn select_kernel() -> MontyKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => adx_kernel,
        X86Backend::Bmi2 => bmi2_kernel,
        X86Backend::Adx | X86Backend::Baseline => fallback_kernel,
    }
}

#[inline]
pub fn selected_kernel() -> MontyKernel {
    *KERNEL.get_or_init(select_kernel)
}
