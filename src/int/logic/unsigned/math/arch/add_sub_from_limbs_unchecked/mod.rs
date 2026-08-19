//! Shared-source simultaneous addition and subtraction kernel.

#![allow(
    unsafe_code,
    reason = "Raw limb kernels and x86-64 ADX assembly require unsafe operations"
)]

select_arch_kernel! {
    function: add_sub_from_limbs_unchecked;
    kernel: AddSubFromKernel;
    surface: selector;
    backends: [];
    x86_64: [fallback, avx2, adx];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
    test_backends: [
        add_sub_from_limbs_unchecked_avx2_test => x86_64_avx2,
    ];
}
