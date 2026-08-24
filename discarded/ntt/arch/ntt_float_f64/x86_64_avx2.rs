//! AVX2+FMA 4-lane Vectorized 50-bit floating-point Harvey NTT butterfly kernels.

#![allow(
    unsafe_code,
    reason = "Target feature AVX2/FMA intrinsics natively require unsafe code"
)]
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Standard mathematical notation for FFT butterflies"
)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _MM_HINT_T0, _mm_prefetch,
    _mm256_add_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_round_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
};

use super::{NttFloatKernels, mulmod_scalar, radix4_dif_float_one, radix4_dit_float_one};

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reduce_to_pm1n_avx2(a: __m256d, n: __m256d, ninv: __m256d) -> __m256d {
    let q = _mm256_round_pd(
        _mm256_mul_pd(a, ninv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    _mm256_fnmadd_pd(q, n, a)
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn mulmod_avx2(a: __m256d, b: __m256d, n: __m256d, ninv: __m256d) -> __m256d {
    let h = _mm256_mul_pd(a, b);
    let q = _mm256_round_pd(
        _mm256_mul_pd(h, ninv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let l = _mm256_fmsub_pd(a, b, h);
    let rem_h = _mm256_fnmadd_pd(q, n, h);
    _mm256_add_pd(rem_h, l)
}

/// Vectorized 2x unrolled 8-lane Radix-4 decimation-in-frequency step.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `3 * quarter_len` `f64` elements.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn radix4_dif_float_avx2(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes avx2/fma and validates buffer bounds.
    unsafe {
        let vec_p = _mm256_set1_pd(prime);
        let vec_pinv = _mm256_set1_pd(pinv);
        let q2 = quarter_len.wrapping_mul(2);
        let q3 = quarter_len.wrapping_mul(3);

        let mut index = 0_usize;
        while index.wrapping_add(8) <= quarter_len {
            _mm_prefetch(
                values.wrapping_add(index.wrapping_add(32)).cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(quarter_len.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(q2.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(q3.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );

            let a0 = _mm256_loadu_pd(values.add(index));
            let a1 = _mm256_loadu_pd(values.add(index.wrapping_add(4)));
            let b0 = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index)));
            let b1 = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index).wrapping_add(4)));
            let c0 = _mm256_loadu_pd(values.add(q2.wrapping_add(index)));
            let c1 = _mm256_loadu_pd(values.add(q2.wrapping_add(index).wrapping_add(4)));
            let d0 = _mm256_loadu_pd(values.add(q3.wrapping_add(index)));
            let d1 = _mm256_loadu_pd(values.add(q3.wrapping_add(index).wrapping_add(4)));

            let tw0_0 = _mm256_loadu_pd(twiddles.add(index));
            let tw0_1 = _mm256_loadu_pd(twiddles.add(index.wrapping_add(4)));
            let tw1_0 = _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index)));
            let tw1_1 =
                _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index).wrapping_add(4)));
            let second_tw0 = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index)));
            let second_tw1 = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index).wrapping_add(4)));

            let red_a0 = reduce_to_pm1n_avx2(a0, vec_p, vec_pinv);
            let red_a1 = reduce_to_pm1n_avx2(a1, vec_p, vec_pinv);

            let low_sum0 = reduce_to_pm1n_avx2(_mm256_add_pd(red_a0, c0), vec_p, vec_pinv);
            let low_sum1 = reduce_to_pm1n_avx2(_mm256_add_pd(red_a1, c1), vec_p, vec_pinv);

            let low_diff0 = mulmod_avx2(_mm256_sub_pd(red_a0, c0), tw0_0, vec_p, vec_pinv);
            let low_diff1 = mulmod_avx2(_mm256_sub_pd(red_a1, c1), tw0_1, vec_p, vec_pinv);

            let high_sum0 = reduce_to_pm1n_avx2(_mm256_add_pd(b0, d0), vec_p, vec_pinv);
            let high_sum1 = reduce_to_pm1n_avx2(_mm256_add_pd(b1, d1), vec_p, vec_pinv);

            let high_diff0 = mulmod_avx2(_mm256_sub_pd(b0, d0), tw1_0, vec_p, vec_pinv);
            let high_diff1 = mulmod_avx2(_mm256_sub_pd(b1, d1), tw1_1, vec_p, vec_pinv);

            let out0_0 = _mm256_add_pd(low_sum0, high_sum0);
            let out0_1 = _mm256_add_pd(low_sum1, high_sum1);

            let out1_0 = mulmod_avx2(
                _mm256_sub_pd(low_sum0, high_sum0),
                second_tw0,
                vec_p,
                vec_pinv,
            );
            let out1_1 = mulmod_avx2(
                _mm256_sub_pd(low_sum1, high_sum1),
                second_tw1,
                vec_p,
                vec_pinv,
            );

            let out2_0 = _mm256_add_pd(low_diff0, high_diff0);
            let out2_1 = _mm256_add_pd(low_diff1, high_diff1);

            let out3_0 = mulmod_avx2(
                _mm256_sub_pd(low_diff0, high_diff0),
                second_tw0,
                vec_p,
                vec_pinv,
            );
            let out3_1 = mulmod_avx2(
                _mm256_sub_pd(low_diff1, high_diff1),
                second_tw1,
                vec_p,
                vec_pinv,
            );

            _mm256_storeu_pd(values.add(index), out0_0);
            _mm256_storeu_pd(values.add(index.wrapping_add(4)), out0_1);
            _mm256_storeu_pd(values.add(quarter_len.wrapping_add(index)), out1_0);
            _mm256_storeu_pd(
                values.add(quarter_len.wrapping_add(index).wrapping_add(4)),
                out1_1,
            );
            _mm256_storeu_pd(values.add(q2.wrapping_add(index)), out2_0);
            _mm256_storeu_pd(values.add(q2.wrapping_add(index).wrapping_add(4)), out2_1);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index)), out3_0);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index).wrapping_add(4)), out3_1);

            index = index.wrapping_add(8);
        }

        while index.wrapping_add(4) <= quarter_len {
            let a_raw = _mm256_loadu_pd(values.add(index));
            let b = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index)));
            let c = _mm256_loadu_pd(values.add(q2.wrapping_add(index)));
            let d = _mm256_loadu_pd(values.add(q3.wrapping_add(index)));
            let tw0 = _mm256_loadu_pd(twiddles.add(index));
            let tw1 = _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index)));
            let second_twiddle = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index)));

            let a = reduce_to_pm1n_avx2(a_raw, vec_p, vec_pinv);

            let low_sum = reduce_to_pm1n_avx2(_mm256_add_pd(a, c), vec_p, vec_pinv);
            let low_diff = mulmod_avx2(_mm256_sub_pd(a, c), tw0, vec_p, vec_pinv);
            let high_sum = reduce_to_pm1n_avx2(_mm256_add_pd(b, d), vec_p, vec_pinv);
            let high_diff = mulmod_avx2(_mm256_sub_pd(b, d), tw1, vec_p, vec_pinv);

            let out0 = _mm256_add_pd(low_sum, high_sum);
            let out1 = mulmod_avx2(
                _mm256_sub_pd(low_sum, high_sum),
                second_twiddle,
                vec_p,
                vec_pinv,
            );
            let out2 = _mm256_add_pd(low_diff, high_diff);
            let out3 = mulmod_avx2(
                _mm256_sub_pd(low_diff, high_diff),
                second_twiddle,
                vec_p,
                vec_pinv,
            );

            _mm256_storeu_pd(values.add(index), out0);
            _mm256_storeu_pd(values.add(quarter_len.wrapping_add(index)), out1);
            _mm256_storeu_pd(values.add(q2.wrapping_add(index)), out2);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index)), out3);

            index = index.wrapping_add(4);
        }

        while index < quarter_len {
            radix4_dif_float_one(values, twiddles, index, quarter_len, prime, pinv);
            index = index.wrapping_add(1);
        }
    }
}

/// Vectorized 2x unrolled 8-lane Radix-4 decimation-in-time step.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `3 * quarter_len` `f64` elements.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn radix4_dit_float_avx2(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes avx2/fma and validates buffer bounds.
    unsafe {
        let vec_p = _mm256_set1_pd(prime);
        let vec_pinv = _mm256_set1_pd(pinv);
        let q2 = quarter_len.wrapping_mul(2);
        let q3 = quarter_len.wrapping_mul(3);

        let mut index = 0_usize;
        while index.wrapping_add(8) <= quarter_len {
            _mm_prefetch(
                values.wrapping_add(index.wrapping_add(32)).cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(quarter_len.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(q2.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                values
                    .wrapping_add(q3.wrapping_add(index).wrapping_add(32))
                    .cast::<i8>(),
                _MM_HINT_T0,
            );

            let a0 = _mm256_loadu_pd(values.add(index));
            let a1 = _mm256_loadu_pd(values.add(index.wrapping_add(4)));
            let b0 = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index)));
            let b1 = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index).wrapping_add(4)));
            let c0 = _mm256_loadu_pd(values.add(q2.wrapping_add(index)));
            let c1 = _mm256_loadu_pd(values.add(q2.wrapping_add(index).wrapping_add(4)));
            let d0 = _mm256_loadu_pd(values.add(q3.wrapping_add(index)));
            let d1 = _mm256_loadu_pd(values.add(q3.wrapping_add(index).wrapping_add(4)));

            let tw0_0 = _mm256_loadu_pd(twiddles.add(index));
            let tw0_1 = _mm256_loadu_pd(twiddles.add(index.wrapping_add(4)));
            let tw1_0 = _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index)));
            let tw1_1 =
                _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index).wrapping_add(4)));
            let second_tw0 = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index)));
            let second_tw1 = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index).wrapping_add(4)));

            let low_twiddled0 = mulmod_avx2(b0, second_tw0, vec_p, vec_pinv);
            let low_twiddled1 = mulmod_avx2(b1, second_tw1, vec_p, vec_pinv);

            let high_twiddled0 = mulmod_avx2(d0, second_tw0, vec_p, vec_pinv);
            let high_twiddled1 = mulmod_avx2(d1, second_tw1, vec_p, vec_pinv);

            let low_sum0 = reduce_to_pm1n_avx2(_mm256_add_pd(a0, low_twiddled0), vec_p, vec_pinv);
            let low_sum1 = reduce_to_pm1n_avx2(_mm256_add_pd(a1, low_twiddled1), vec_p, vec_pinv);

            let low_diff0 = reduce_to_pm1n_avx2(_mm256_sub_pd(a0, low_twiddled0), vec_p, vec_pinv);
            let low_diff1 = reduce_to_pm1n_avx2(_mm256_sub_pd(a1, low_twiddled1), vec_p, vec_pinv);

            let high_sum0 = reduce_to_pm1n_avx2(_mm256_add_pd(c0, high_twiddled0), vec_p, vec_pinv);
            let high_sum1 = reduce_to_pm1n_avx2(_mm256_add_pd(c1, high_twiddled1), vec_p, vec_pinv);

            let high_diff0 =
                reduce_to_pm1n_avx2(_mm256_sub_pd(c0, high_twiddled0), vec_p, vec_pinv);
            let high_diff1 =
                reduce_to_pm1n_avx2(_mm256_sub_pd(c1, high_twiddled1), vec_p, vec_pinv);

            let high_sum_twiddled0 = mulmod_avx2(high_sum0, tw0_0, vec_p, vec_pinv);
            let high_sum_twiddled1 = mulmod_avx2(high_sum1, tw0_1, vec_p, vec_pinv);

            let high_diff_twiddled0 = mulmod_avx2(high_diff0, tw1_0, vec_p, vec_pinv);
            let high_diff_twiddled1 = mulmod_avx2(high_diff1, tw1_1, vec_p, vec_pinv);

            let out0_0 = _mm256_add_pd(low_sum0, high_sum_twiddled0);
            let out0_1 = _mm256_add_pd(low_sum1, high_sum_twiddled1);

            let out1_0 = _mm256_add_pd(low_diff0, high_diff_twiddled0);
            let out1_1 = _mm256_add_pd(low_diff1, high_diff_twiddled1);

            let out2_0 = _mm256_sub_pd(low_sum0, high_sum_twiddled0);
            let out2_1 = _mm256_sub_pd(low_sum1, high_sum_twiddled1);

            let out3_0 = _mm256_sub_pd(low_diff0, high_diff_twiddled0);
            let out3_1 = _mm256_sub_pd(low_diff1, high_diff_twiddled1);

            _mm256_storeu_pd(values.add(index), out0_0);
            _mm256_storeu_pd(values.add(index.wrapping_add(4)), out0_1);
            _mm256_storeu_pd(values.add(quarter_len.wrapping_add(index)), out1_0);
            _mm256_storeu_pd(
                values.add(quarter_len.wrapping_add(index).wrapping_add(4)),
                out1_1,
            );
            _mm256_storeu_pd(values.add(q2.wrapping_add(index)), out2_0);
            _mm256_storeu_pd(values.add(q2.wrapping_add(index).wrapping_add(4)), out2_1);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index)), out3_0);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index).wrapping_add(4)), out3_1);

            index = index.wrapping_add(8);
        }

        while index.wrapping_add(4) <= quarter_len {
            let a = _mm256_loadu_pd(values.add(index));
            let b = _mm256_loadu_pd(values.add(quarter_len.wrapping_add(index)));
            let c = _mm256_loadu_pd(values.add(q2.wrapping_add(index)));
            let d = _mm256_loadu_pd(values.add(q3.wrapping_add(index)));
            let tw0 = _mm256_loadu_pd(twiddles.add(index));
            let tw1 = _mm256_loadu_pd(twiddles.add(quarter_len.wrapping_add(index)));
            let second_twiddle = _mm256_loadu_pd(twiddles.add(q2.wrapping_add(index)));

            let low_twiddled = mulmod_avx2(b, second_twiddle, vec_p, vec_pinv);
            let high_twiddled = mulmod_avx2(d, second_twiddle, vec_p, vec_pinv);

            let low_sum = reduce_to_pm1n_avx2(_mm256_add_pd(a, low_twiddled), vec_p, vec_pinv);
            let low_diff = reduce_to_pm1n_avx2(_mm256_sub_pd(a, low_twiddled), vec_p, vec_pinv);
            let high_sum = reduce_to_pm1n_avx2(_mm256_add_pd(c, high_twiddled), vec_p, vec_pinv);
            let high_diff = reduce_to_pm1n_avx2(_mm256_sub_pd(c, high_twiddled), vec_p, vec_pinv);

            let high_sum_twiddled = mulmod_avx2(high_sum, tw0, vec_p, vec_pinv);
            let high_diff_twiddled = mulmod_avx2(high_diff, tw1, vec_p, vec_pinv);

            let out0 = _mm256_add_pd(low_sum, high_sum_twiddled);
            let out1 = _mm256_add_pd(low_diff, high_diff_twiddled);
            let out2 = _mm256_sub_pd(low_sum, high_sum_twiddled);
            let out3 = _mm256_sub_pd(low_diff, high_diff_twiddled);

            _mm256_storeu_pd(values.add(index), out0);
            _mm256_storeu_pd(values.add(quarter_len.wrapping_add(index)), out1);
            _mm256_storeu_pd(values.add(q2.wrapping_add(index)), out2);
            _mm256_storeu_pd(values.add(q3.wrapping_add(index)), out3);

            index = index.wrapping_add(4);
        }

        while index < quarter_len {
            radix4_dit_float_one(values, twiddles, index, quarter_len, prime, pinv);
            index = index.wrapping_add(1);
        }
    }
}

/// AVX2-accelerated 4-lane pointwise frequency-domain multiplication.
///
/// # Safety
/// `a` and `b` are valid for reading `len` elements, and `a` is valid for writing `len` elements.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn pointwise_mul_float_avx2(
    a: *mut f64,
    b: *const f64,
    len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes avx2/fma and validates buffer bounds.
    unsafe {
        let vec_p = _mm256_set1_pd(prime);
        let vec_pinv = _mm256_set1_pd(pinv);
        let mut i = 0_usize;
        while i.wrapping_add(8) <= len {
            let va0 = _mm256_loadu_pd(a.add(i));
            let va1 = _mm256_loadu_pd(a.add(i.wrapping_add(4)));
            let vb0 = _mm256_loadu_pd(b.add(i));
            let vb1 = _mm256_loadu_pd(b.add(i.wrapping_add(4)));
            let res0 = mulmod_avx2(va0, vb0, vec_p, vec_pinv);
            let res1 = mulmod_avx2(va1, vb1, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res0);
            _mm256_storeu_pd(a.add(i.wrapping_add(4)), res1);
            i = i.wrapping_add(8);
        }
        while i.wrapping_add(4) <= len {
            let va = _mm256_loadu_pd(a.add(i));
            let vb = _mm256_loadu_pd(b.add(i));
            let res = mulmod_avx2(va, vb, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res);
            i = i.wrapping_add(4);
        }
        while i < len {
            let va = *a.add(i);
            let vb = *b.add(i);
            *a.add(i) = mulmod_scalar(va, vb, prime, pinv);
            i = i.wrapping_add(1);
        }
    }
}

/// AVX2-accelerated 4-lane pointwise frequency-domain squaring.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn pointwise_sqr_float_avx2(a: *mut f64, len: usize, prime: f64, pinv: f64) {
    // SAFETY: caller establishes avx2/fma and validates buffer bounds.
    unsafe {
        let vec_p = _mm256_set1_pd(prime);
        let vec_pinv = _mm256_set1_pd(pinv);
        let mut i = 0_usize;
        while i.wrapping_add(8) <= len {
            let va0 = _mm256_loadu_pd(a.add(i));
            let va1 = _mm256_loadu_pd(a.add(i.wrapping_add(4)));
            let res0 = mulmod_avx2(va0, va0, vec_p, vec_pinv);
            let res1 = mulmod_avx2(va1, va1, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res0);
            _mm256_storeu_pd(a.add(i.wrapping_add(4)), res1);
            i = i.wrapping_add(8);
        }
        while i.wrapping_add(4) <= len {
            let va = _mm256_loadu_pd(a.add(i));
            let res = mulmod_avx2(va, va, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res);
            i = i.wrapping_add(4);
        }
        while i < len {
            let va = *a.add(i);
            *a.add(i) = mulmod_scalar(va, va, prime, pinv);
            i = i.wrapping_add(1);
        }
    }
}

/// AVX2-accelerated 4-lane frequency-domain scaling by `inv_n`.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn scale_float_avx2(a: *mut f64, len: usize, inv_n: f64, prime: f64, pinv: f64) {
    // SAFETY: caller establishes avx2/fma and validates buffer bounds.
    unsafe {
        let vec_p = _mm256_set1_pd(prime);
        let vec_pinv = _mm256_set1_pd(pinv);
        let vec_inv = _mm256_set1_pd(inv_n);
        let mut i = 0_usize;
        while i.wrapping_add(8) <= len {
            let va0 = _mm256_loadu_pd(a.add(i));
            let va1 = _mm256_loadu_pd(a.add(i.wrapping_add(4)));
            let res0 = mulmod_avx2(va0, vec_inv, vec_p, vec_pinv);
            let res1 = mulmod_avx2(va1, vec_inv, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res0);
            _mm256_storeu_pd(a.add(i.wrapping_add(4)), res1);
            i = i.wrapping_add(8);
        }
        while i.wrapping_add(4) <= len {
            let va = _mm256_loadu_pd(a.add(i));
            let res = mulmod_avx2(va, vec_inv, vec_p, vec_pinv);
            _mm256_storeu_pd(a.add(i), res);
            i = i.wrapping_add(4);
        }
        while i < len {
            let va = *a.add(i);
            *a.add(i) = mulmod_scalar(va, inv_n, prime, pinv);
            i = i.wrapping_add(1);
        }
    }
}

#[inline]
pub fn ntt_float_f64() -> NttFloatKernels {
    NttFloatKernels {
        radix4_dif: radix4_dif_float_avx2,
        radix4_dit: radix4_dit_float_avx2,
        pointwise_mul: pointwise_mul_float_avx2,
        pointwise_sqr: pointwise_sqr_float_avx2,
        scale: scale_float_avx2,
    }
}
