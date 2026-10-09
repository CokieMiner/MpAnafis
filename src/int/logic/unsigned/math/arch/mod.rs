//! Limb-kernel facade, target selection, and CPU-feature detection.
//!
//! Operation modules select a compiled backend or a cached runtime backend.
//! Arithmetic algorithms call `ArchKernels` without target-specific branches.

use super::{DoubleLimb, LIMB_BITS, Limb};

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
#[cfg(not(target_pointer_width = "16"))]
mod lshift_overlapping_unchecked;
mod lshift_unchecked;
mod monty_redc_unchecked;
mod mul_2_limbs_unchecked;
mod mul_basecase_unchecked;
mod propagate_borrow_unchecked;
mod propagate_carry_unchecked;
mod rshift_into_unchecked;
mod rshift_unchecked;
mod shifts;
mod signatures;
mod sqr_basecase_unchecked;
mod sub_limbs_3_unchecked;
mod sub_limbs_unchecked;
mod sub_mul_limbs_unchecked;
#[cfg(not(target_pointer_width = "16"))]
mod sub_shifted_high_limbs_unchecked;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64"
))]
mod x86_runtime;

pub use add_limbs_3_unchecked::add_limbs_3_unchecked;
pub use add_limbs_unchecked::add_limbs_unchecked;
#[cfg(not(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
)))]
pub use add_mul_2_limbs_unchecked::kernel as selected_add_mul_2_kernel;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use add_mul_2_limbs_unchecked::{
    add_mul_2_limbs_bmi2_backend, add_mul_2_limbs_vanilla_backend,
};
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use add_mul_limbs_unchecked::{
    add_mul_limbs_adx_backend, add_mul_limbs_bmi2_backend, add_mul_limbs_vanilla_backend,
};
pub use add_mul_limbs_unchecked::{add_mul_limbs_unchecked, kernel as selected_add_mul_kernel};
pub use add_reverse_sub_limbs_unchecked::add_reverse_sub_limbs_unchecked;
#[cfg(not(target_pointer_width = "16"))]
pub use add_sub_from_limbs_unchecked::kernel as selected_add_sub_from_kernel;
pub use add_sub_limbs_unchecked::add_sub_limbs_unchecked;
pub use add_two_limbs_unchecked::add_two_limbs_unchecked;
pub use divrem_1_unchecked::divrem_1_unchecked;
pub use kernels::ArchKernels;
pub use lshift_into_unchecked::{
    kernel as selected_lshift_into_kernel, small_kernel as selected_lshift_into_small_kernel,
};
#[cfg(not(target_pointer_width = "16"))]
pub use lshift_overlapping_unchecked::kernel as selected_lshift_overlapping_kernel;
pub use lshift_unchecked::kernel as selected_lshift_kernel;
pub use monty_redc_unchecked::kernel as selected_monty_redc_kernel;
#[cfg(not(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
)))]
pub use mul_2_limbs_unchecked::kernel as selected_mul_2_kernel;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use mul_2_limbs_unchecked::{mul_2_limbs_bmi2_backend, mul_2_limbs_vanilla_backend};
pub use mul_basecase_unchecked::{
    mul_2x2_portable_unchecked, mul_3x3_portable_unchecked, mul_4x4_unchecked, mul_8x8_unchecked,
    mul_basecase_unchecked,
};
pub use propagate_borrow_unchecked::propagate_borrow_unchecked;
pub use propagate_carry_unchecked::propagate_carry_unchecked;
pub use rshift_into_unchecked::{
    kernel as selected_rshift_into_kernel, small_kernel as selected_rshift_into_small_kernel,
};
pub use rshift_unchecked::kernel as selected_rshift_kernel;
#[cfg(not(target_pointer_width = "16"))]
pub use signatures::AddSubFromKernel;
#[cfg(not(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
)))]
pub use signatures::{AddMul2Kernel, Mul2Kernel};
pub use signatures::{
    AddMulKernel, LshiftIntoKernel, LshiftKernel, MontyKernel, RshiftIntoKernel, RshiftKernel,
    SubMulKernel,
};
#[cfg(not(target_pointer_width = "16"))]
pub use signatures::{LshiftOverlappingKernel, SubShiftedHighKernel};
pub use sqr_basecase_unchecked::sqr_basecase_unchecked;
pub use sub_limbs_3_unchecked::sub_limbs_3_unchecked;
pub use sub_limbs_unchecked::sub_limbs_unchecked;
pub use sub_mul_limbs_unchecked::{kernel as selected_sub_mul_kernel, sub_mul_limbs_unchecked};
#[cfg(not(target_pointer_width = "16"))]
pub use sub_shifted_high_limbs_unchecked::kernel as selected_sub_shifted_high_kernel;
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(all(target_feature = "adx", target_feature = "bmi2"))
))]
pub use x86_runtime::{X86Backend, selected_x86_backend};
#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64"
))]
pub use x86_runtime::{X86SimdTier, selected_x86_simd_tier};

#[cfg(test)]
mod tests;
