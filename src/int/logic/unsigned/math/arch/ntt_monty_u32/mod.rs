//! Architecture-specific 31-bit Montgomery NTT kernels.

#![allow(
    unsafe_code,
    reason = "Target feature AVX2 intrinsics natively require unsafe code"
)]

mod scalar;

#[cfg(any(
    feature = "_internal-tune",
    all(target_arch = "aarch64", target_pointer_width = "64"),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "x86_64", feature = "std", not(miri))
))]
pub use scalar::{radix4_dif_one, radix4_dit_one};

#[cfg(any(
    test,
    feature = "_internal-tune",
    all(
        not(all(target_arch = "aarch64", target_pointer_width = "64")),
        not(all(target_arch = "x86_64", target_feature = "avx2"))
    )
))]
pub use scalar::{radix4_dif_scalar, radix4_dit_scalar};

select_arch_kernel! {
    function: ntt_monty_u32;
    kernel: NttMontyKernel;
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
