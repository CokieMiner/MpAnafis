//! Architecture-specific 50-bit floating-point Harvey NTT kernels.

#![allow(
    unsafe_code,
    reason = "Target feature AVX2/NEON intrinsics natively require unsafe code"
)]

mod scalar;

#[cfg(test)]
mod tests;

select_arch_kernel! {
    function: ntt_float_f64;
    kernel: NttFloatKernel;
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

#[cfg(test)]
pub use scalar::reduce_to_pm1n_scalar;
#[cfg(any(
    test,
    all(
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        any(feature = "std", target_feature = "avx2")
    ),
    all(not(miri), target_arch = "aarch64", target_pointer_width = "64")
))]
pub use scalar::{mulmod_scalar, radix4_dif_float_one, radix4_dit_float_one};
#[cfg(any(
    test,
    miri,
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    all(
        not(miri),
        target_arch = "x86_64",
        target_pointer_width = "64",
        not(target_feature = "avx2")
    )
))]
pub use scalar::{
    pointwise_mul_float_scalar, pointwise_sqr_float_scalar, radix4_dif_float_scalar,
    radix4_dit_float_scalar, scale_float_scalar,
};
