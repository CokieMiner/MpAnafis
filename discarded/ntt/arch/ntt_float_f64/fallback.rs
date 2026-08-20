//! Portable fallback 50-bit floating-point Harvey NTT kernel provider.

use super::{
    NttFloatKernels, pointwise_mul_float_scalar, pointwise_sqr_float_scalar,
    radix4_dif_float_scalar, radix4_dit_float_scalar, scale_float_scalar,
};

#[inline]
pub fn ntt_float_f64() -> NttFloatKernels {
    NttFloatKernels {
        radix4_dif: radix4_dif_float_scalar,
        radix4_dit: radix4_dit_float_scalar,
        pointwise_mul: pointwise_mul_float_scalar,
        pointwise_sqr: pointwise_sqr_float_scalar,
        scale: scale_float_scalar,
    }
}
