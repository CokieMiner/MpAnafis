//! Two-lane NEON kernels with fused product-error recovery.

#![expect(
    unsafe_code,
    reason = "The module gate establishes NEON; kernel contracts bound all initialized intrinsic spans"
)]

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vmulq_f64, vnegq_f64,
    vrndnq_f64, vst1q_f64, vsubq_f64,
};

use super::Scalar;

vector_backend!(
    Neon,
    float64x2_t,
    2,
    "neon",
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vaddq_f64,
    vsubq_f64,
    vmulq_f64,
    |a, b, c| vfmaq_f64(vnegq_f64(c), a, b),
    |a, b, c| vfmsq_f64(c, a, b),
    vrndnq_f64
);
