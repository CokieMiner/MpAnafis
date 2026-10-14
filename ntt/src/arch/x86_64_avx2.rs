//! Four-lane AVX2 and FMA kernels.

#![expect(
    unsafe_code,
    reason = "The selector validates AVX2 and FMA before calling span-checked intrinsic kernels"
)]

use core::arch::x86_64::{
    __m256d, _mm256_add_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_round_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
};

use super::Scalar;

vector_backend!(
    Avx2,
    __m256d,
    4,
    "avx2,fma",
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_set1_pd,
    _mm256_add_pd,
    _mm256_sub_pd,
    _mm256_mul_pd,
    _mm256_fmsub_pd,
    _mm256_fnmadd_pd,
    |a| _mm256_round_pd::<8>(a)
);
