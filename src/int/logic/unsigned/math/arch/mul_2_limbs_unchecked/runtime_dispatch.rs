//! Runtime CPU-feature selection for the write-only `mul_2` kernel on `x86_64`.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, selected_x86_backend, x86_64::mul_2_limbs_unchecked as mul_2_limbs_vanilla,
    x86_64_bmi2::mul_2_limbs_unchecked as mul_2_limbs_bmi2,
};

type Mul2Fn = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb);

static KERNEL: OnceLock<Mul2Fn> = OnceLock::new();

fn select_kernel() -> Mul2Fn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Bmi2 => mul_2_limbs_bmi2,
        X86Backend::Adx | X86Backend::Baseline => mul_2_limbs_vanilla,
    }
}

#[inline]
pub fn selected_kernel() -> Mul2Fn {
    *KERNEL.get_or_init(select_kernel)
}
