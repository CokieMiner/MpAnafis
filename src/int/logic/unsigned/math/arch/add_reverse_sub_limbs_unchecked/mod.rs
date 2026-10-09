//! Simultaneous addition and reverse-subtraction kernel.
//!
//! With base `B = 2^LIMB_BITS`, the outputs satisfy
//! `sum_new + carry * B^len = sum_old + difference_old` and
//! `difference_new - borrow * B^len = difference_old - sum_old`.
//! Both returned flags are binary.

#![expect(
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

#[cfg(test)]
mod tests;
