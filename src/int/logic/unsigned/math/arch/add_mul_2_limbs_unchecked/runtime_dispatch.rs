//! Runtime CPU-feature dispatch for `add_mul_2` on `x86_64`.
//!
//! The shared architecture selector resolves CPU features once; this module
//! maps that stable feature level to the operation-specific function pointer.

use std::sync::OnceLock;

use super::{
    Limb, X86Backend, selected_x86_backend,
    x86_64::add_mul_2_limbs_unchecked as add_mul_2_limbs_vanilla,
    x86_64_bmi2::add_mul_2_limbs_unchecked as add_mul_2_limbs_bmi2,
};

type Mul2Fn = unsafe fn(*mut Limb, *const Limb, usize, Limb, Limb) -> (Limb, Limb);

struct Dispatch {
    kernel: Mul2Fn,
}

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

fn select_dispatch() -> Dispatch {
    match selected_x86_backend() {
        X86Backend::AdxBmi2 | X86Backend::Bmi2 => Dispatch {
            kernel: add_mul_2_limbs_bmi2,
        },
        X86Backend::Adx | X86Backend::Baseline => Dispatch {
            kernel: add_mul_2_limbs_vanilla,
        },
    }
}

#[inline]
fn selected_dispatch() -> &'static Dispatch {
    DISPATCH.get_or_init(select_dispatch)
}

#[inline]
pub fn selected_kernel() -> Mul2Fn {
    selected_dispatch().kernel
}
