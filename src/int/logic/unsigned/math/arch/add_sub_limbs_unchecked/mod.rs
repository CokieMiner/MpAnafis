//! Simultaneous fixed-width addition and subtraction kernel.

#![allow(
    unsafe_code,
    reason = "Raw limb kernels and x86-64 ADX assembly require unsafe operations"
)]

mod availability;

use super::{ArchKernels, Limb};

select_arch_kernel! {
    function: add_sub_limbs_unchecked;
    surface: direct;
    backends: [];
    x86_64: [adx];
    powerpc64: [];
    special_coverage: [];
    fallback_imports: [];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
))]
pub use runtime_dispatch::fast_add_sub_limbs_available as runtime_fast_add_sub_limbs_available;
