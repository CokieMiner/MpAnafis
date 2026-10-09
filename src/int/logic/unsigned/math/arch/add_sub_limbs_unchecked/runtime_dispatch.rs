//! Runtime dispatch for the x86-64 simultaneous add/subtract kernel.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, adx_kernel, fallback_kernel, selected_x86_backend,
};

type KernelFn = unsafe fn(*mut Limb, *mut Limb, usize) -> (Limb, Limb);

struct Dispatch {
    kernel: KernelFn,
    has_independent_carries: bool,
}

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

/// Dispatch simultaneous addition and subtraction to the selected backend.
///
/// # Safety
///
/// - For nonzero `len`, both pointers must be aligned and valid for reads and
///   writes of `len` initialized limbs, with byte spans at most `isize::MAX`.
/// - The two spans must not overlap.
#[inline]
pub unsafe fn add_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    let kernel = DISPATCH.get_or_init(select_dispatch).kernel;
    // SAFETY: the caller establishes both spans; selection guarantees any CPU
    // feature required by the chosen backend.
    unsafe { kernel(sum, difference, len) }
}

/// Returns whether runtime selection uses independent ADX carry chains.
#[inline]
pub fn fast_add_sub_limbs_available() -> bool {
    DISPATCH.get_or_init(select_dispatch).has_independent_carries
}

fn select_dispatch() -> Dispatch {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Adx => Dispatch {
            kernel: adx_kernel,
            has_independent_carries: true,
        },
        X86Backend::Bmi2 | X86Backend::Baseline => Dispatch {
            kernel: fallback_kernel,
            has_independent_carries: false,
        },
    }
}
