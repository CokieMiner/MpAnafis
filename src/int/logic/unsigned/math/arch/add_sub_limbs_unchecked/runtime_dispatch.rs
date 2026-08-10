//! Runtime dispatch for the x86-64 simultaneous add/subtract kernel.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, fallback::add_sub_limbs_unchecked as fallback_kernel, selected_x86_backend,
    x86_64_adx::add_sub_limbs_unchecked as adx_kernel,
};

type KernelFn = unsafe fn(*mut Limb, *mut Limb, usize) -> (Limb, Limb);

struct Dispatch {
    kernel: KernelFn,
    has_independent_carries: bool,
}

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

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

#[inline]
fn selected_dispatch() -> &'static Dispatch {
    DISPATCH.get_or_init(select_dispatch)
}

/// Whether runtime selection can use independent ADX carry chains.
#[inline]
pub fn fast_add_sub_limbs_available() -> bool {
    selected_dispatch().has_independent_carries
}

/// Dispatch simultaneous addition and subtraction to the selected backend.
///
/// # Safety
///
/// - Both pointers must be valid for reads and writes of `len` limbs.
/// - The two spans must not overlap.
#[inline]
pub unsafe fn add_sub_limbs_unchecked(
    sum: *mut Limb,
    difference: *mut Limb,
    len: usize,
) -> (Limb, Limb) {
    let kernel = selected_dispatch().kernel;
    // SAFETY: the caller establishes both spans; selection guarantees any CPU
    // feature required by the chosen backend.
    unsafe { kernel(sum, difference, len) }
}
