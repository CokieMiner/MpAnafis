//! Runtime dispatch for the x86-64 dual-addition kernel.
//!
//! The shared architecture selector resolves CPU features once; this module
//! maps ADX-capable levels to the independent-carry-chain kernel.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, fallback::add_two_limbs_unchecked as fallback_add_two, selected_x86_backend,
    x86_64_adx::add_two_limbs_unchecked as adx_add_two,
};

type AddTwoFn = unsafe fn(*mut Limb, *const Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb);

static KERNEL: OnceLock<AddTwoFn> = OnceLock::new();

fn select_kernel() -> AddTwoFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Adx => adx_add_two,
        X86Backend::Bmi2 | X86Backend::Baseline => fallback_add_two,
    }
}

/// Dispatch two independent additions to the selected backend.
///
/// # Safety
///
/// - Every pointer must cover `len` readable limbs.
/// - Both destination pointers must cover `len` writable limbs.
/// - No destination span may overlap any other span.
#[inline]
pub unsafe fn add_two_limbs_unchecked(
    dst_a: *mut Limb,
    src_a: *const Limb,
    dst_b: *mut Limb,
    src_b: *const Limb,
    len: usize,
) -> (Limb, Limb) {
    let kernel = *KERNEL.get_or_init(select_kernel);
    // SAFETY: the caller establishes all span invariants; selection proves
    // any CPU feature required by the chosen backend.
    unsafe { kernel(dst_a, src_a, dst_b, src_b, len) }
}
