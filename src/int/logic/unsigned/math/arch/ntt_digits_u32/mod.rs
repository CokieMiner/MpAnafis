//! Architecture-specific packing of 64-bit limbs into 16-bit NTT digits.

#![allow(
    unsafe_code,
    reason = "The selected packing kernels operate on caller-validated raw spans"
)]

mod scalar;

#[cfg(all(
    not(all(target_arch = "aarch64", target_pointer_width = "64")),
    not(all(target_arch = "x86_64", target_feature = "avx2"))
))]
pub use scalar::limbs_to_digits_16_scalar;

select_arch_kernel! {
    function: ntt_digits_u32;
    kernel: NttDigitsKernel;
    surface: selector;
    backends: [
        aarch64 => all(not(miri), target_arch = "aarch64", target_pointer_width = "64"),
    ];
    x86_64: [sse2, avx2];
    powerpc64: [];
    special_coverage: [
        all(target_arch = "x86_64", target_pointer_width = "64"),
    ];
    fallback_imports: [];
    test_backends: [];
}
