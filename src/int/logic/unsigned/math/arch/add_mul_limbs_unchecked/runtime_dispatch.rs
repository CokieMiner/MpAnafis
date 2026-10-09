//! Cached x86-64 backend selection for scalar multiply-add rows.
//!
//! The shared architecture selector resolves CPU features once; this module
//! maps that stable feature level to the operation-specific function pointer.
//!
use std::sync::OnceLock;

use super::{
    AddMulKernel, Limb, X86Backend, add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend,
    add_mul_limbs_vanilla_backend, selected_x86_backend,
};

static KERNEL: OnceLock<AddMulKernel> = OnceLock::new();

/// Returns the cached backend, initializing it from the detected CPU features.
#[inline]
pub fn selected_kernel() -> AddMulKernel {
    *KERNEL.get_or_init(select_kernel)
}

/// Multiply `src` by one limb and accumulate the product into `dst`.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[inline]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    KERNEL.get().map_or_else(
        || {
            // SAFETY: the caller establishes both disjoint spans; the cold
            // executor selects a backend before invoking the same operands.
            unsafe { initialize_and_execute(dst, src, len, scalar) }
        },
        |kernel| {
            // SAFETY: the caller establishes both disjoint spans. KERNEL holds
            // only a backend whose prerequisites were proved by select_kernel.
            unsafe { kernel(dst, src, len, scalar) }
        },
    )
}

/// Select and execute the first row without keeping its operands live across
/// an initialization call in the warm entry point.
///
/// # Safety
///
/// Nonempty spans must satisfy the caller's aligned, initialized, disjoint
/// limb-span contract, including a writable destination.
#[cold]
#[inline(never)]
unsafe fn initialize_and_execute(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let kernel = selected_kernel();
    // SAFETY: the caller establishes the initialized, disjoint spans, and
    // selected_kernel proves the selected backend's CPU prerequisites.
    unsafe { kernel(dst, src, len, scalar) }
}

fn select_kernel() -> AddMulKernel {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 => add_mul_limbs_adx_backend,
        X86Backend::Bmi2 => add_mul_limbs_bmi2_backend,
        X86Backend::Adx | X86Backend::Baseline => add_mul_limbs_vanilla_backend,
    }
}
