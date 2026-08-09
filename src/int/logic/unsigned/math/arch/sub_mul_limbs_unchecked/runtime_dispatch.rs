//! Runtime CPU-feature dispatch for `sub_mul_limbs_unchecked` on `x86_64`.
//!
//! The shared architecture selector resolves CPU features once; this module
//! maps that stable feature level to the operation-specific function pointer.
//!
//! ## Testing override (debug builds only)
//!
//! ```bash
//! MP_ANAFIS_TEST_BACKEND=bmi2 cargo test test_
//! MP_ANAFIS_TEST_BACKEND=vanilla cargo test test_
//! ```

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, selected_x86_backend,
    x86_64::sub_mul_limbs_unchecked as sub_mul_limbs_vanilla,
    x86_64_adx::sub_mul_limbs_unchecked as sub_mul_limbs_adx,
    x86_64_bmi2::sub_mul_limbs_unchecked as sub_mul_limbs_bmi2,
};

type SubMulFn = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> (Limb, Limb);

static KERNEL: OnceLock<SubMulFn> = OnceLock::new();

fn select_kernel() -> SubMulFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => sub_mul_limbs_adx,
        X86Backend::Bmi2 => sub_mul_limbs_bmi2,
        X86Backend::Adx | X86Backend::Baseline => sub_mul_limbs_vanilla,
    }
}

#[inline]
pub fn selected_kernel() -> SubMulFn {
    *KERNEL.get_or_init(select_kernel)
}

/// Multiply `src` by one limb and subtract the product from `dst`.
///
/// # Safety
///
/// `src` and `dst` must each cover `len` limbs, and their spans must not
/// overlap. The selected backend may require CPU features proved by the shared
/// runtime selector.
#[inline]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let kernel = selected_kernel();
    // SAFETY: the caller establishes both spans; backend selection proves any
    // additional CPU feature requirement.
    unsafe { kernel(dst, src, len, scalar) }
}
