//! CPU selection and exact floating-point residue kernels.

#[cfg(any(
    all(
        not(miri),
        target_arch = "x86_64",
        any(
            feature = "std",
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma")
        )
    ),
    all(not(miri), target_arch = "aarch64", target_feature = "neon")
))]
#[macro_use]
mod vector;
#[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
mod aarch64;
mod fma;
mod scalar;
mod selector;
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(
        feature = "std",
        all(
            not(target_feature = "avx512f"),
            target_feature = "avx2",
            target_feature = "fma"
        )
    )
))]
mod x86_64_avx2;
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(feature = "std", target_feature = "avx512f")
))]
mod x86_64_avx512;

#[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64::Neon;
pub use fma::Fma;
pub use scalar::Scalar;
pub use selector::{NttFloatKernels, Radix4Kernel};
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(
        feature = "std",
        all(
            not(target_feature = "avx512f"),
            target_feature = "avx2",
            target_feature = "fma"
        )
    )
))]
pub use x86_64_avx2::Avx2;
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(feature = "std", target_feature = "avx512f")
))]
pub use x86_64_avx512::Avx512;

#[cfg(test)]
mod tests;
