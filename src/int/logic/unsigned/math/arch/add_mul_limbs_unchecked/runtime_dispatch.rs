//! Runtime CPU feature dispatch for `multiplication` on `x86_64`.
//!
//! The shared architecture selector resolves CPU features once; this module
//! maps that stable feature level to the operation-specific function pointer.
//!
//! ## Testing override (debug builds only)
//!
//! ```bash
//! ARBI_TEST_BACKEND=adx cargo test test_
//! ARBI_TEST_BACKEND=bmi2 cargo test test_
//! ARBI_TEST_BACKEND=vanilla cargo test test_
//! ```

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, selected_x86_backend,
    x86_64::add_mul_limbs_unchecked as add_mul_limbs_vanilla,
    x86_64_adx::add_mul_limbs_unchecked as add_mul_limbs_adx,
    x86_64_bmi2::add_mul_limbs_unchecked as add_mul_limbs_bmi2,
};

type MulFn = unsafe fn(*mut Limb, *const Limb, usize, Limb) -> Limb;

static KERNEL: OnceLock<MulFn> = OnceLock::new();

fn select_kernel() -> MulFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => add_mul_limbs_adx,
        X86Backend::Bmi2 => add_mul_limbs_bmi2,
        X86Backend::Adx | X86Backend::Baseline => add_mul_limbs_vanilla,
    }
}

#[inline]
pub fn selected_kernel() -> MulFn {
    *KERNEL.get_or_init(select_kernel)
}

/// Multiply `src` by one limb and accumulate the product into `dst`.
///
/// # Safety
///
/// `src` and `dst` must each cover `len` limbs, and their spans must not
/// overlap. The selected backend may require CPU features proved by the shared
/// runtime selector.
#[inline]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let kernel = selected_kernel();
    // SAFETY: the caller establishes both spans; backend selection proves any
    // additional CPU feature requirement.
    unsafe { kernel(dst, src, len, scalar) }
}
