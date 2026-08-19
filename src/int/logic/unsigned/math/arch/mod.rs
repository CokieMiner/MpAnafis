//! Architecture-specific limb-kernel boundary.
//!
//! Each operation module owns its target `cfg` selection, optional runtime CPU
//! dispatch, inline assembly, and portable fallback. Callers in the arithmetic
//! tower see only the architecture-neutral `ArchKernels` methods or
//! explicitly re-exported kernel functions; target selection must not leak
//! into `math/mul` or other algorithm modules.
//!
//! Fallbacks use the loop shape that best expresses the kernel's dependency
//! chain to LLVM. Hot fallbacks may be inlined when call-boundary removal is
//! measured or required for constant propagation, but neither raw pointers nor
//! `inline(always)` are blanket requirements.

#![allow(
    unsafe_code,
    reason = "Hardware inline assembly natively requires unsafe code"
)]

use super::{DoubleLimb, LIMB_BITS, Limb};

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(
        target_feature = "adx",
        target_feature = "bmi2",
        target_feature = "avx2"
    ))
))]
use x86_runtime::selected_x86_backend;

#[macro_use]
mod backend_providers;
#[macro_use]
mod x86_selectors;
#[macro_use]
mod kernel_selection;
mod add_limbs_3_unchecked;
mod add_limbs_unchecked;
mod add_mul_2_limbs_unchecked;
mod add_mul_limbs_unchecked;
mod add_reverse_sub_limbs_unchecked;
#[cfg(not(target_pointer_width = "16"))]
mod add_sub_from_limbs_unchecked;
mod add_sub_limbs_unchecked;
mod add_two_limbs_unchecked;
mod divrem_1_unchecked;
mod kernels;
mod lshift_into_unchecked;
mod lshift_unchecked;
mod monty_redc_unchecked;
mod mul_2_limbs_unchecked;
mod mul_basecase_unchecked;
mod ntt_digits_u32;
mod ntt_monty_u32;
mod propagate_borrow_unchecked;
mod propagate_carry_unchecked;
mod rshift_into_unchecked;
mod rshift_unchecked;
mod sub_limbs_3_unchecked;
mod sub_limbs_unchecked;
mod sub_mul_limbs_unchecked;
#[cfg(not(target_pointer_width = "16"))]
mod sub_shifted_high_limbs_unchecked;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(
        target_feature = "adx",
        target_feature = "bmi2",
        target_feature = "avx2"
    ))
))]
mod x86_runtime;

#[cfg(test)]
mod tests;

pub use kernels::ArchKernels;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(
        target_feature = "adx",
        target_feature = "bmi2",
        target_feature = "avx2"
    ))
))]
pub use x86_runtime::X86Backend;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "avx2")
))]
pub use x86_runtime::{X86SimdTier, selected_x86_simd_tier};
