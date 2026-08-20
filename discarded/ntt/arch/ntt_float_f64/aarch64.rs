//! `AArch64` NEON Vectorized 50-bit floating-point Harvey NTT butterfly kernels.

#![allow(
    unsafe_code,
    reason = "Target feature NEON intrinsics natively require unsafe code"
)]
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "Standard mathematical notation for FFT butterflies"
)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vrndnq_f64, vst1q_f64,
    vsubq_f64,
};

use super::{mulmod_scalar, radix4_dif_float_one, radix4_dit_float_one, NttFloatKernels};

#[inline]
#[target_feature(enable = "neon")]
pub fn reduce_to_pm1n_neon(a: float64x2_t, n: float64x2_t, ninv: float64x2_t) -> float64x2_t {
    let q = vrndnq_f64(vmulq_f64(a, ninv));
    let neg_n = vsubq_f64(vdupq_n_f64(0.0), n);
    vfmaq_f64(a, q, neg_n)
}

#[inline]
#[target_feature(enable = "neon")]
pub fn mulmod_neon(
    a: float64x2_t,
    b: float64x2_t,
    n: float64x2_t,
    ninv: float64x2_t,
) -> float64x2_t {
    let h = vmulq_f64(a, b);
    let q = vrndnq_f64(vmulq_f64(h, ninv));
    let neg_h = vsubq_f64(vdupq_n_f64(0.0), h);
    let l = vfmaq_f64(neg_h, a, b);
    let neg_n = vsubq_f64(vdupq_n_f64(0.0), n);
    let rem_h = vfmaq_f64(h, q, neg_n);
    vaddq_f64(rem_h, l)
}

/// Vectorized 2x unrolled 4-lane Radix-4 decimation-in-frequency step for `AArch64`.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `3 * quarter_len` `f64` elements.
#[target_feature(enable = "neon")]
pub unsafe fn radix4_dif_float_neon(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes neon and validates buffer spans.
    unsafe {
        let vec_p = vdupq_n_f64(prime);
        let vec_pinv = vdupq_n_f64(pinv);
        let q2 = quarter_len.wrapping_mul(2);
        let q3 = quarter_len.wrapping_mul(3);

        let mut index = 0_usize;
        while index.wrapping_add(4) <= quarter_len {
            let a0 = vld1q_f64(values.add(index));
            let a1 = vld1q_f64(values.add(index.wrapping_add(2)));
            let b0 = vld1q_f64(values.add(quarter_len.wrapping_add(index)));
            let b1 = vld1q_f64(values.add(quarter_len.wrapping_add(index).wrapping_add(2)));
            let c0 = vld1q_f64(values.add(q2.wrapping_add(index)));
            let c1 = vld1q_f64(values.add(q2.wrapping_add(index).wrapping_add(2)));
            let d0 = vld1q_f64(values.add(q3.wrapping_add(index)));
            let d1 = vld1q_f64(values.add(q3.wrapping_add(index).wrapping_add(2)));

            let tw0_0 = vld1q_f64(twiddles.add(index));
            let tw0_1 = vld1q_f64(twiddles.add(index.wrapping_add(2)));
            let tw1_0 = vld1q_f64(twiddles.add(quarter_len.wrapping_add(index)));
            let tw1_1 = vld1q_f64(twiddles.add(quarter_len.wrapping_add(index).wrapping_add(2)));
            let second_tw0 = vld1q_f64(twiddles.add(q2.wrapping_add(index)));
            let second_tw1 = vld1q_f64(twiddles.add(q2.wrapping_add(index).wrapping_add(2)));

            let red_a0 = reduce_to_pm1n_neon(a0, vec_p, vec_pinv);
            let red_a1 = reduce_to_pm1n_neon(a1, vec_p, vec_pinv);

            let low_sum0 = reduce_to_pm1n_neon(vaddq_f64(red_a0, c0), vec_p, vec_pinv);
            let low_sum1 = reduce_to_pm1n_neon(vaddq_f64(red_a1, c1), vec_p, vec_pinv);

            let low_diff0 = mulmod_neon(vsubq_f64(red_a0, c0), tw0_0, vec_p, vec_pinv);
            let low_diff1 = mulmod_neon(vsubq_f64(red_a1, c1), tw0_1, vec_p, vec_pinv);

            let high_sum0 = reduce_to_pm1n_neon(vaddq_f64(b0, d0), vec_p, vec_pinv);
            let high_sum1 = reduce_to_pm1n_neon(vaddq_f64(b1, d1), vec_p, vec_pinv);

            let high_diff0 = mulmod_neon(vsubq_f64(b0, d0), tw1_0, vec_p, vec_pinv);
            let high_diff1 = mulmod_neon(vsubq_f64(b1, d1), tw1_1, vec_p, vec_pinv);

            let out0_0 = vaddq_f64(low_sum0, high_sum0);
            let out0_1 = vaddq_f64(low_sum1, high_sum1);

            let out1_0 = mulmod_neon(vsubq_f64(low_sum0, high_sum0), second_tw0, vec_p, vec_pinv);
            let out1_1 = mulmod_neon(vsubq_f64(low_sum1, high_sum1), second_tw1, vec_p, vec_pinv);

            let out2_0 = vaddq_f64(low_diff0, high_diff0);
            let out2_1 = vaddq_f64(low_diff1, high_diff1);

            let out3_0 = mulmod_neon(
                vsubq_f64(low_diff0, high_diff0),
                second_tw0,
                vec_p,
                vec_pinv,
            );
            let out3_1 = mulmod_neon(
                vsubq_f64(low_diff1, high_diff1),
                second_tw1,
                vec_p,
                vec_pinv,
            );

            vst1q_f64(values.add(index), out0_0);
            vst1q_f64(values.add(index.wrapping_add(2)), out0_1);
            vst1q_f64(values.add(quarter_len.wrapping_add(index)), out1_0);
            vst1q_f64(
                values.add(quarter_len.wrapping_add(index).wrapping_add(2)),
                out1_1,
            );
            vst1q_f64(values.add(q2.wrapping_add(index)), out2_0);
            vst1q_f64(values.add(q2.wrapping_add(index).wrapping_add(2)), out2_1);
            vst1q_f64(values.add(q3.wrapping_add(index)), out3_0);
            vst1q_f64(values.add(q3.wrapping_add(index).wrapping_add(2)), out3_1);

            index = index.wrapping_add(4);
        }

        while index < quarter_len {
            radix4_dif_float_one(values, twiddles, index, quarter_len, prime, pinv);
            index = index.wrapping_add(1);
        }
    }
}

/// Vectorized 2x unrolled 4-lane Radix-4 decimation-in-time step for `AArch64`.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `3 * quarter_len` `f64` elements.
#[target_feature(enable = "neon")]
pub unsafe fn radix4_dit_float_neon(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes neon and validates buffer spans.
    unsafe {
        let vec_p = vdupq_n_f64(prime);
        let vec_pinv = vdupq_n_f64(pinv);
        let q2 = quarter_len.wrapping_mul(2);
        let q3 = quarter_len.wrapping_mul(3);

        let mut index = 0_usize;
        while index.wrapping_add(4) <= quarter_len {
            let a0 = vld1q_f64(values.add(index));
            let a1 = vld1q_f64(values.add(index.wrapping_add(2)));
            let b0 = vld1q_f64(values.add(quarter_len.wrapping_add(index)));
            let b1 = vld1q_f64(values.add(quarter_len.wrapping_add(index).wrapping_add(2)));
            let c0 = vld1q_f64(values.add(q2.wrapping_add(index)));
            let c1 = vld1q_f64(values.add(q2.wrapping_add(index).wrapping_add(2)));
            let d0 = vld1q_f64(values.add(q3.wrapping_add(index)));
            let d1 = vld1q_f64(values.add(q3.wrapping_add(index).wrapping_add(2)));

            let tw0_0 = vld1q_f64(twiddles.add(index));
            let tw0_1 = vld1q_f64(twiddles.add(index.wrapping_add(2)));
            let tw1_0 = vld1q_f64(twiddles.add(quarter_len.wrapping_add(index)));
            let tw1_1 = vld1q_f64(twiddles.add(quarter_len.wrapping_add(index).wrapping_add(2)));
            let second_tw0 = vld1q_f64(twiddles.add(q2.wrapping_add(index)));
            let second_tw1 = vld1q_f64(twiddles.add(q2.wrapping_add(index).wrapping_add(2)));

            let low_twiddled0 = mulmod_neon(b0, second_tw0, vec_p, vec_pinv);
            let low_twiddled1 = mulmod_neon(b1, second_tw1, vec_p, vec_pinv);

            let high_twiddled0 = mulmod_neon(d0, second_tw0, vec_p, vec_pinv);
            let high_twiddled1 = mulmod_neon(d1, second_tw1, vec_p, vec_pinv);

            let low_sum0 = reduce_to_pm1n_neon(vaddq_f64(a0, low_twiddled0), vec_p, vec_pinv);
            let low_sum1 = reduce_to_pm1n_neon(vaddq_f64(a1, low_twiddled1), vec_p, vec_pinv);

            let low_diff0 = reduce_to_pm1n_neon(vsubq_f64(a0, low_twiddled0), vec_p, vec_pinv);
            let low_diff1 = reduce_to_pm1n_neon(vsubq_f64(a1, low_twiddled1), vec_p, vec_pinv);

            let high_sum0 = reduce_to_pm1n_neon(vaddq_f64(c0, high_twiddled0), vec_p, vec_pinv);
            let high_sum1 = reduce_to_pm1n_neon(vaddq_f64(c1, high_twiddled1), vec_p, vec_pinv);

            let high_diff0 = reduce_to_pm1n_neon(vsubq_f64(c0, high_twiddled0), vec_p, vec_pinv);
            let high_diff1 = reduce_to_pm1n_neon(vsubq_f64(c1, high_twiddled1), vec_p, vec_pinv);

            let high_sum_twiddled0 = mulmod_neon(high_sum0, tw0_0, vec_p, vec_pinv);
            let high_sum_twiddled1 = mulmod_neon(high_sum1, tw0_1, vec_p, vec_pinv);

            let high_diff_twiddled0 = mulmod_neon(high_diff0, tw1_0, vec_p, vec_pinv);
            let high_diff_twiddled1 = mulmod_neon(high_diff1, tw1_1, vec_p, vec_pinv);

            let out0_0 = vaddq_f64(low_sum0, high_sum_twiddled0);
            let out0_1 = vaddq_f64(low_sum1, high_sum_twiddled1);

            let out1_0 = vaddq_f64(low_diff0, high_diff0);
            let out1_1 = vaddq_f64(low_diff1, high_diff1);

            let out2_0 = vsubq_f64(low_sum0, high_sum_twiddled0);
            let out2_1 = vsubq_f64(low_sum1, high_sum_twiddled1);

            let out3_0 = vsubq_f64(low_diff0, high_diff0);
            let out3_1 = vsubq_f64(low_diff1, high_diff1);

            vst1q_f64(values.add(index), out0_0);
            vst1q_f64(values.add(index.wrapping_add(2)), out0_1);
            vst1q_f64(values.add(quarter_len.wrapping_add(index)), out1_0);
            vst1q_f64(
                values.add(quarter_len.wrapping_add(index).wrapping_add(2)),
                out1_1,
            );
            vst1q_f64(values.add(q2.wrapping_add(index)), out2_0);
            vst1q_f64(values.add(q2.wrapping_add(index).wrapping_add(2)), out2_1);
            vst1q_f64(values.add(q3.wrapping_add(index)), out3_0);
            vst1q_f64(values.add(q3.wrapping_add(index).wrapping_add(2)), out3_1);

            index = index.wrapping_add(4);
        }

        while index < quarter_len {
            radix4_dit_float_one(values, twiddles, index, quarter_len, prime, pinv);
            index = index.wrapping_add(1);
        }
    }
}

/// NEON-accelerated 2-lane pointwise frequency-domain multiplication.
///
/// # Safety
/// `a` and `b` are valid for reading `len` elements, and `a` is valid for writing `len` elements.
#[target_feature(enable = "neon")]
pub unsafe fn pointwise_mul_float_neon(
    a: *mut f64,
    b: *const f64,
    len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes neon and validates buffer spans.
    unsafe {
        let vec_p = vdupq_n_f64(prime);
        let vec_pinv = vdupq_n_f64(pinv);
        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let va0 = vld1q_f64(a.add(i));
            let va1 = vld1q_f64(a.add(i.wrapping_add(2)));
            let vb0 = vld1q_f64(b.add(i));
            let vb1 = vld1q_f64(b.add(i.wrapping_add(2)));
            let res0 = mulmod_neon(va0, vb0, vec_p, vec_pinv);
            let res1 = mulmod_neon(va1, vb1, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res0);
            vst1q_f64(a.add(i.wrapping_add(2)), res1);
            i = i.wrapping_add(4);
        }
        while i.wrapping_add(2) <= len {
            let va = vld1q_f64(a.add(i));
            let vb = vld1q_f64(b.add(i));
            let res = mulmod_neon(va, vb, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res);
            i = i.wrapping_add(2);
        }
        while i < len {
            let va = *a.add(i);
            let vb = *b.add(i);
            *a.add(i) = mulmod_scalar(va, vb, prime, pinv);
            i = i.wrapping_add(1);
        }
    }
}

/// NEON-accelerated 2-lane pointwise frequency-domain squaring.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[target_feature(enable = "neon")]
pub unsafe fn pointwise_sqr_float_neon(
    a: *mut f64,
    len: usize,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes neon and validates buffer spans.
    unsafe {
        let vec_p = vdupq_n_f64(prime);
        let vec_pinv = vdupq_n_f64(pinv);
        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let va0 = vld1q_f64(a.add(i));
            let va1 = vld1q_f64(a.add(i.wrapping_add(2)));
            let res0 = mulmod_neon(va0, va0, vec_p, vec_pinv);
            let res1 = mulmod_neon(va1, va1, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res0);
            vst1q_f64(a.add(i.wrapping_add(2)), res1);
            i = i.wrapping_add(4);
        }
        while i.wrapping_add(2) <= len {
            let va = vld1q_f64(a.add(i));
            let res = mulmod_neon(va, va, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res);
            i = i.wrapping_add(2);
        }
        while i < len {
            let va = *a.add(i);
            *a.add(i) = mulmod_scalar(va, va, prime, pinv);
            i = i.wrapping_add(1);
        }
    }
}

/// NEON-accelerated 2-lane frequency-domain scaling by `inv_n`.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[target_feature(enable = "neon")]
pub unsafe fn scale_float_neon(
    a: *mut f64,
    len: usize,
    inv_n: f64,
    prime: f64,
    pinv: f64,
) {
    // SAFETY: caller establishes neon and validates buffer spans.
    unsafe {
        let vec_p = vdupq_n_f64(prime);
        let vec_pinv = vdupq_n_f64(pinv);
        let vec_inv = vdupq_n_f64(inv_n);
        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let va0 = vld1q_f64(a.add(i));
            let va1 = vld1q_f64(a.add(i.wrapping_add(2)));
            let res0 = mulmod_neon(va0, vec_inv, vec_p, vec_pinv);
            let res1 = mulmod_neon(va1, vec_inv, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res0);
            vst1q_f64(a.add(i.wrapping_add(2)), res1);
            i = i.wrapping_add(4);
        }
        while i.wrapping_add(2) <= len {
            let va = vld1q_f64(a.add(i));
            let res = mulmod_neon(va, vec_inv, vec_p, vec_pinv);
            vst1q_f64(a.add(i), res);
            i = i.wrapping_add(2);
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
        radix4_dif: radix4_dif_float_neon,
        radix4_dit: radix4_dit_float_neon,
        pointwise_mul: pointwise_mul_float_neon,
        pointwise_sqr: pointwise_sqr_float_neon,
        scale: scale_float_neon,
    }
}
