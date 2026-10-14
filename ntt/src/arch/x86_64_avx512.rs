//! Eight-lane AVX-512F kernels; no AVX-512DQ operations are required.

#![expect(
    unsafe_code,
    reason = "The selector validates AVX-512F before calling span-checked intrinsic kernels"
)]

use core::arch::x86_64::{
    __m512d, _mm512_add_pd, _mm512_fmsub_pd, _mm512_fnmadd_pd, _mm512_loadu_pd, _mm512_mul_pd,
    _mm512_roundscale_pd, _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
};

use super::Scalar;

vector_backend!(
    Avx512,
    __m512d,
    8,
    "avx512f",
    _mm512_loadu_pd,
    _mm512_storeu_pd,
    _mm512_set1_pd,
    _mm512_add_pd,
    _mm512_sub_pd,
    _mm512_mul_pd,
    _mm512_fmsub_pd,
    _mm512_fnmadd_pd,
    |a| _mm512_roundscale_pd::<8>(a)
);
