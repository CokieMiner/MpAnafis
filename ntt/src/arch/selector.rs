//! One-time selection of an admissible residue kernel table.

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
use super::Avx2;
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(feature = "std", target_feature = "avx512f")
))]
use super::Avx512;
#[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
use super::Neon;
#[cfg(not(any(
    all(not(miri), target_arch = "aarch64", target_feature = "neon"),
    all(
        not(miri),
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx512f"
    ),
    all(
        not(miri),
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
use super::Scalar;

/// Raw radix-4 stage interface.
pub type Radix4Kernel = unsafe fn(*mut f64, *const f64, usize, usize, usize, f64, f64);
type PointwiseKernel = unsafe fn(*mut f64, *const f64, usize, f64, f64);
type ScaleKernel = unsafe fn(*mut f64, usize, f64, f64, f64);

/// ISA-validated function table retained by a prepared workspace.
#[derive(Clone, Copy, Debug)]
pub struct NttFloatKernels {
    pub dif: Radix4Kernel,
    pub dit: Radix4Kernel,
    pub multiply: PointwiseKernel,
    pub square: PointwiseKernel,
    pub scale: ScaleKernel,
}

impl NttFloatKernels {
    /// Resolves CPU features once at the complete-operation boundary.
    #[must_use]
    pub fn get() -> Self {
        #[cfg(all(not(miri), target_arch = "x86_64", feature = "std"))]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Self {
                    dif: Avx512::radix4::<false>,
                    dit: Avx512::radix4::<true>,
                    multiply: Avx512::pointwise::<false>,
                    square: Avx512::pointwise::<true>,
                    scale: Avx512::scale,
                };
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Self {
                    dif: Avx2::radix4::<false>,
                    dit: Avx2::radix4::<true>,
                    multiply: Avx2::pointwise::<false>,
                    square: Avx2::pointwise::<true>,
                    scale: Avx2::scale,
                };
            }
        }
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            not(feature = "std"),
            target_feature = "avx512f"
        ))]
        {
            Self {
                dif: Avx512::radix4::<false>,
                dit: Avx512::radix4::<true>,
                multiply: Avx512::pointwise::<false>,
                square: Avx512::pointwise::<true>,
                scale: Avx512::scale,
            }
        }
        #[cfg(all(
            not(miri),
            target_arch = "x86_64",
            not(feature = "std"),
            not(target_feature = "avx512f"),
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            Self {
                dif: Avx2::radix4::<false>,
                dit: Avx2::radix4::<true>,
                multiply: Avx2::pointwise::<false>,
                square: Avx2::pointwise::<true>,
                scale: Avx2::scale,
            }
        }
        #[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
        {
            Self {
                dif: Neon::radix4::<false>,
                dit: Neon::radix4::<true>,
                multiply: Neon::pointwise::<false>,
                square: Neon::pointwise::<true>,
                scale: Neon::scale,
            }
        }
        #[cfg(not(any(
            all(not(miri), target_arch = "aarch64", target_feature = "neon"),
            all(
                not(miri),
                target_arch = "x86_64",
                not(feature = "std"),
                target_feature = "avx512f"
            ),
            all(
                not(miri),
                target_arch = "x86_64",
                not(feature = "std"),
                target_feature = "avx2",
                target_feature = "fma"
            )
        )))]
        Self {
            dif: Scalar::radix4::<false>,
            dit: Scalar::radix4::<true>,
            multiply: Scalar::pointwise::<false>,
            square: Scalar::pointwise::<true>,
            scale: Scalar::scale,
        }
    }
}
