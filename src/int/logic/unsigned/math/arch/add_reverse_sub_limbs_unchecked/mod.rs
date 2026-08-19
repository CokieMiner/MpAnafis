//! Simultaneous addition and reverse-subtraction kernel.

#![allow(
    unsafe_code,
    reason = "Raw limb kernels and x86-64 ADX assembly require unsafe operations"
)]

use super::Limb;

select_arch_kernel! {
    function: add_reverse_sub_limbs_unchecked;
    surface: direct;
    backends: [];
    x86_64: [adx];
    powerpc64: [];
    special_coverage: [];
    fallback_imports: [];
}
