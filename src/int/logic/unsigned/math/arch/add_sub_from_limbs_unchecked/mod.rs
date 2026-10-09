//! Shared-source simultaneous addition and subtraction kernel.
//!
//! With base `B = 2^LIMB_BITS`, the outputs satisfy
//! `sum_new + carry * B^len = sum_old + source` and
//! `difference_new - borrow * B^len = sum_old - source`.
//! Both returned flags are binary.

#![expect(
    unsafe_code,
    reason = "Raw limb kernels and x86-64 ADX assembly require unsafe operations"
)]

select_arch_kernel! {
    function: add_sub_from_limbs_unchecked;
    kernel: AddSubFromKernel;
    surface: selector;
    backends: [];
    x86_64: [fallback, adx];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
}

#[cfg(all(
    feature = "std",
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    not(target_feature = "adx")
))]
pub use self::{
    fallback::add_sub_from_limbs_unchecked as fallback_kernel,
    x86_64_adx::add_sub_from_limbs_unchecked as adx_kernel,
};

#[cfg(test)]
mod tests;
