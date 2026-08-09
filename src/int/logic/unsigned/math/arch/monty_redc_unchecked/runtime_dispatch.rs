//! Runtime CPU-feature dispatch for `monty_redc_step_unchecked` on `x86_64`.
//!
//! The shared architecture selector resolves CPU features once. Baseline x86-64
//! uses the portable dual-carry kernel; BMI2 and ADX+BMI2 select their dedicated
//! assembly kernels.

use std::sync::OnceLock;

use super::{
    MontyKernel, X86Backend,
    fallback::monty_redc_step_unchecked as monty_redc_step_fallback, selected_x86_backend,
    x86_64_adx::monty_redc_step_unchecked as monty_redc_step_adx,
    x86_64_bmi2::monty_redc_step_unchecked as monty_redc_step_bmi2,
};

static KERNEL: OnceLock<MontyKernel> = OnceLock::new();

fn select_kernel() -> MontyKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => monty_redc_step_adx,
        X86Backend::Bmi2 => monty_redc_step_bmi2,
        X86Backend::Adx | X86Backend::Baseline => monty_redc_step_fallback,
    }
}

#[inline]
pub fn selected_kernel() -> MontyKernel {
    *KERNEL.get_or_init(select_kernel)
}
