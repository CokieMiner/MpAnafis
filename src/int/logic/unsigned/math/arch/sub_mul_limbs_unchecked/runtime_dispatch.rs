//! Cached CPU selection for single-limb multiply-subtract.

use std::sync::OnceLock;

use super::{
    Limb, SubMulKernel, X86Backend, adx_kernel, bmi2_kernel, fallback_kernel,
    selected_x86_backend,
};

static KERNEL: OnceLock<SubMulKernel> = OnceLock::new();

/// Subtracts `src * scalar` from `dst` and returns product carry and borrow.
///
/// # Safety
///
/// Both pointers must cover `len` aligned, initialized limbs in disjoint spans;
/// `dst` requires exclusive access. Each span's byte length must fit in `isize`.
/// Zero length permits null pointers.
#[inline]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let kernel = selected_kernel();
    // SAFETY: the caller supplies the disjoint initialized spans and byte
    // bounds. The cached selector establishes BMI2 and ADX when required.
    unsafe { kernel(dst, src, len, scalar) }
}

/// Returns the backend selected from the available CPU features.
#[inline]
pub fn selected_kernel() -> SubMulKernel {
    *KERNEL.get_or_init(select_kernel)
}

fn select_kernel() -> SubMulKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => adx_kernel,
        X86Backend::Bmi2 => bmi2_kernel,
        X86Backend::Adx | X86Backend::Baseline => fallback_kernel,
    }
}
