//! Runtime dispatch for the x86-64 simultaneous add/reverse-subtract kernel.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, fallback::add_reverse_sub_limbs_unchecked as fallback_kernel,
    selected_x86_backend, x86_64_adx::add_reverse_sub_limbs_unchecked as adx_kernel,
};

type KernelFn = unsafe fn(*mut Limb, *mut Limb, usize) -> (Limb, Limb);

static KERNEL: OnceLock<KernelFn> = OnceLock::new();

fn select_kernel() -> KernelFn {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Adx => adx_kernel,
        X86Backend::Bmi2 | X86Backend::Baseline => fallback_kernel,
    }
}

/// Dispatch simultaneous addition and reverse subtraction to the selected backend.
///
/// # Safety
///
/// - Both pointers must be valid for reads and writes of `len` limbs.
/// - The two spans must not overlap.
#[inline]
pub unsafe fn add_reverse_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    let kernel = *KERNEL.get_or_init(select_kernel);
    // SAFETY: the caller establishes both spans; selection guarantees any CPU
    // feature required by the chosen backend.
    unsafe { kernel(sum, difference, len) }
}
