//! Fused arithmetic for finite integer-valued NTT operands.
//!
//! Software evaluation uses exact signed integer arithmetic. The NTT bounds
//! every product and addend below `2^104`, so their exact sum fits i128 and
//! conversion to binary64 performs the sole rounding.

#![cfg_attr(
    any(
        not(feature = "std"),
        all(not(miri), target_arch = "x86_64", target_feature = "fma"),
        all(not(miri), target_arch = "aarch64", target_feature = "neon")
    ),
    expect(
        unsafe_code,
        reason = "Compiled ISA gates or bounded integer-valued inputs prove the fused operation"
    )
)]

#[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::vfmad_f64;
#[cfg(all(not(miri), target_arch = "x86_64", target_feature = "fma"))]
use core::arch::x86_64::{_mm_cvtsd_f64, _mm_fmadd_sd, _mm_set_sd};

/// Single-rounding arithmetic under the transform's integer bounds.
#[derive(Debug)]
pub struct Fma;

impl Fma {
    /// Returns `RN(a*b+c)` for integer operands with magnitudes below `2^104`.
    ///
    /// The exact product and sum must also lie below `2^104` in magnitude.
    #[inline]
    pub fn mul_add(a: f64, b: f64, c: f64) -> f64 {
        #[cfg(all(not(miri), target_arch = "x86_64", target_feature = "fma"))]
        {
            // SAFETY: the compiled fma feature establishes CPU and OS support.
            unsafe { _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c))) }
        }
        #[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
        {
            vfmad_f64(c, a, b)
        }
        #[cfg(not(any(
            all(not(miri), target_arch = "x86_64", target_feature = "fma"),
            all(not(miri), target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            #[cfg(feature = "std")]
            {
                a.mul_add(b, c)
            }
            #[cfg(not(feature = "std"))]
            {
                // SAFETY: finite integer operands, their product, and their
                // sum have magnitude below 2^104, strictly within i128.
                let exact = unsafe {
                    a.to_int_unchecked::<i128>()
                        .unchecked_mul(b.to_int_unchecked::<i128>())
                        .unchecked_add(c.to_int_unchecked::<i128>())
                };
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "The exact integer sum is rounded once to binary64, as required by fused arithmetic"
                )]
                {
                    exact as f64
                }
            }
        }
    }
}
