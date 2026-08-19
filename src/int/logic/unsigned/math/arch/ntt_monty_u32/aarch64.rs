//! AArch64 NEON 4-lane vectorized Montgomery NTT kernels with Harvey lazy reduction.

use core::arch::aarch64::{
    uint32x2_t, uint32x4_t, vaddq_u32, vbslq_u32, vcgtq_u32, vcombine_u32, vdup_n_u32, vdupq_n_u32,
    vget_high_u32, vget_low_u32, vld1q_u32, vminq_u32, vmlal_u32, vmovn_u64, vmul_u32, vmull_u32,
    vshrq_n_u64, vst1q_u32, vsubq_u32,
};

use super::{NttMontyKernels, radix4_dif_one, radix4_dit_one};

#[allow(
    clippy::inline_always,
    reason = "Critical for peak SIMD loop vectorization and register reuse"
)]
#[inline(always)]
unsafe fn monty_mul_neon_4(
    a: uint32x4_t,
    b: uint32x4_t,
    vec_p_4: uint32x4_t,
    mu_2: uint32x2_t,
    prime_2: uint32x2_t,
) -> uint32x4_t {
    // SAFETY: caller establishes valid 128-bit vector registers.
    unsafe {
        let a_lo = vget_low_u32(a);
        let b_lo = vget_low_u32(b);
        let a_hi = vget_high_u32(a);
        let b_hi = vget_high_u32(b);

        let t_lo = vmull_u32(a_lo, b_lo);
        let t_hi = vmull_u32(a_hi, b_hi);

        let q_lo = vmul_u32(vmovn_u64(t_lo), mu_2);
        let q_hi = vmul_u32(vmovn_u64(t_hi), mu_2);

        let t_plus_qp_lo = vmlal_u32(t_lo, q_lo, prime_2);
        let t_plus_qp_hi = vmlal_u32(t_hi, q_hi, prime_2);

        let m_lo = vmovn_u64(vshrq_n_u64(t_plus_qp_lo, 32));
        let m_hi = vmovn_u64(vshrq_n_u64(t_plus_qp_hi, 32));

        let m = vcombine_u32(m_lo, m_hi);
        let m_sub_p = vsubq_u32(m, vec_p_4);
        vminq_u32(m, m_sub_p)
    }
}

/// Harvey lazy reduction: computes (a * b * R^-1) mod p in [0, 2p) with zero conditional sub.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak SIMD loop vectorization and register reuse"
)]
#[inline(always)]
unsafe fn monty_mul_lazy_neon_4(
    a: uint32x4_t,
    b: uint32x4_t,
    mu_2: uint32x2_t,
    prime_2: uint32x2_t,
) -> uint32x4_t {
    // SAFETY: caller establishes valid 128-bit vector registers.
    unsafe {
        let a_lo = vget_low_u32(a);
        let b_lo = vget_low_u32(b);
        let a_hi = vget_high_u32(a);
        let b_hi = vget_high_u32(b);

        let t_lo = vmull_u32(a_lo, b_lo);
        let t_hi = vmull_u32(a_hi, b_hi);

        let q_lo = vmul_u32(vmovn_u64(t_lo), mu_2);
        let q_hi = vmul_u32(vmovn_u64(t_hi), mu_2);

        let t_plus_qp_lo = vmlal_u32(t_lo, q_lo, prime_2);
        let t_plus_qp_hi = vmlal_u32(t_hi, q_hi, prime_2);

        let m_lo = vmovn_u64(vshrq_n_u64(t_plus_qp_lo, 32));
        let m_hi = vmovn_u64(vshrq_n_u64(t_plus_qp_hi, 32));

        vcombine_u32(m_lo, m_hi)
    }
}

#[inline]
pub unsafe fn monty_mul_slice_neon(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    // SAFETY: caller guarantees valid a, b, and dst pointers for len elements.
    unsafe {
        let vec_p_4 = vdupq_n_u32(prime);
        let prime_2 = vdup_n_u32(prime);
        let mu_2 = vdup_n_u32(neg_inverse);

        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let va_raw = vld1q_u32(a.add(i));
            let vb_raw = vld1q_u32(b.add(i));
            let va = vminq_u32(va_raw, vsubq_u32(va_raw, vec_p_4));
            let vb = vminq_u32(vb_raw, vsubq_u32(vb_raw, vec_p_4));
            let vres = monty_mul_neon_4(va, vb, vec_p_4, mu_2, prime_2);
            vst1q_u32(dst.add(i), vres);
            i = i.wrapping_add(4);
        }
        while i < len {
            let mut x = *a.add(i);
            let mut y = *b.add(i);
            if x >= prime {
                x -= prime;
            }
            if y >= prime {
                y -= prime;
            }
            let t = u64::from(x).wrapping_mul(u64::from(y));
            let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
            let m = (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32)
                as u32;
            let res = if m >= prime { m.wrapping_sub(prime) } else { m };
            *dst.add(i) = res;
            i = i.wrapping_add(1);
        }
    }
}

#[inline]
pub unsafe fn dif_butterfly_neon(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    // SAFETY: caller guarantees valid pointer spans for len elements.
    unsafe {
        let two_p = prime.wrapping_mul(2);
        let vec_2p_4 = vdupq_n_u32(two_p);
        let vec_wrap_4 = vdupq_n_u32(0_u32.wrapping_sub(two_p));
        let prime_2 = vdup_n_u32(prime);
        let mu_2 = vdup_n_u32(neg_inverse);

        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let u = vld1q_u32(low.add(i));
            let v = vld1q_u32(high.add(i));
            let tw = vld1q_u32(twiddles.add(i));

            // The sum can need 33 bits.  On a carry, adding 2^32-2p to the
            // wrapped value gives the exact reduction modulo 2p.
            let sum = vaddq_u32(u, v);
            let sum_sub_2p = vsubq_u32(sum, vec_2p_4);
            let no_carry = vminq_u32(sum, sum_sub_2p);
            let carry = vcgtq_u32(u, sum);
            let carry_result = vaddq_u32(sum, vec_wrap_4);
            let new_u = vbslq_u32(carry, carry_result, no_carry);

            let diff = vsubq_u32(u, v);
            let diff_add_2p = vaddq_u32(diff, vec_2p_4);
            let borrow = vcgtq_u32(v, u);
            let diff = vbslq_u32(borrow, diff_add_2p, diff);
            let new_v = monty_mul_lazy_neon_4(diff, tw, mu_2, prime_2);

            vst1q_u32(low.add(i), new_u);
            vst1q_u32(high.add(i), new_v);
            i = i.wrapping_add(4);
        }
        while i < len {
            let u = *low.add(i);
            let v = *high.add(i);
            let tw = *twiddles.add(i);

            let sum = u64::from(u) + u64::from(v);
            // Both branches are below 2p < 2^32, so these narrowing casts are exact.
            let new_u = if sum >= u64::from(two_p) {
                (sum - u64::from(two_p)) as u32
            } else {
                sum as u32
            };
            let diff = if u >= v {
                u.wrapping_sub(v)
            } else {
                (u64::from(u) + u64::from(two_p) - u64::from(v)) as u32
            };

            let t = u64::from(diff).wrapping_mul(u64::from(tw));
            let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
            let new_v = (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime))))
                .wrapping_shr(32) as u32;

            *low.add(i) = new_u;
            *high.add(i) = new_v;
            i = i.wrapping_add(1);
        }
    }
}

#[inline]
pub unsafe fn dit_butterfly_neon(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    // SAFETY: caller guarantees valid pointer spans for len elements.
    unsafe {
        let two_p = prime.wrapping_mul(2);
        let vec_2p_4 = vdupq_n_u32(two_p);
        let vec_wrap_4 = vdupq_n_u32(0_u32.wrapping_sub(two_p));
        let prime_2 = vdup_n_u32(prime);
        let mu_2 = vdup_n_u32(neg_inverse);

        let mut i = 0_usize;
        while i.wrapping_add(4) <= len {
            let u = vld1q_u32(low.add(i));
            let v = vld1q_u32(high.add(i));
            let tw = vld1q_u32(twiddles.add(i));

            let prod = monty_mul_lazy_neon_4(v, tw, mu_2, prime_2);

            // Widened-by-carry lazy add; see the DIF stage for the bound proof.
            let sum = vaddq_u32(u, prod);
            let sum_sub_2p = vsubq_u32(sum, vec_2p_4);
            let no_carry = vminq_u32(sum, sum_sub_2p);
            let carry = vcgtq_u32(u, sum);
            let carry_result = vaddq_u32(sum, vec_wrap_4);
            let new_u = vbslq_u32(carry, carry_result, no_carry);

            let diff = vsubq_u32(u, prod);
            let diff_add_2p = vaddq_u32(diff, vec_2p_4);
            let borrow = vcgtq_u32(prod, u);
            let new_v = vbslq_u32(borrow, diff_add_2p, diff);

            vst1q_u32(low.add(i), new_u);
            vst1q_u32(high.add(i), new_v);
            i = i.wrapping_add(4);
        }
        while i < len {
            let u = *low.add(i);
            let v = *high.add(i);
            let tw = *twiddles.add(i);

            let t = u64::from(v).wrapping_mul(u64::from(tw));
            let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
            let prod = (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime))))
                .wrapping_shr(32) as u32;

            let sum = u64::from(u) + u64::from(prod);
            // Both branches are below 2p < 2^32, so these narrowing casts are exact.
            let new_u = if sum >= u64::from(two_p) {
                (sum - u64::from(two_p)) as u32
            } else {
                sum as u32
            };
            let new_v = if u >= prod {
                u.wrapping_sub(prod)
            } else {
                (u64::from(u) + u64::from(two_p) - u64::from(prod)) as u32
            };

            *low.add(i) = new_u;
            *high.add(i) = new_v;
            i = i.wrapping_add(1);
        }
    }
}

#[inline]
unsafe fn add_lazy_neon_4(a: uint32x4_t, b: uint32x4_t, two_p: uint32x4_t) -> uint32x4_t {
    // SAFETY: every intrinsic is the AArch64 NEON operation selected by the
    // enclosing target predicate; the arguments are register values.
    unsafe {
        let sum = vaddq_u32(a, b);
        let reduced = vminq_u32(sum, vsubq_u32(sum, two_p));
        let carry = vcgtq_u32(a, sum);
        let wrap = vsubq_u32(vdupq_n_u32(0), two_p);
        vbslq_u32(carry, vaddq_u32(sum, wrap), reduced)
    }
}

#[inline]
unsafe fn sub_lazy_neon_4(a: uint32x4_t, b: uint32x4_t, two_p: uint32x4_t) -> uint32x4_t {
    // SAFETY: every intrinsic is the AArch64 NEON operation selected by the
    // enclosing target predicate; the arguments are register values.
    unsafe {
        let diff = vsubq_u32(a, b);
        let borrow = vcgtq_u32(b, a);
        vbslq_u32(borrow, vaddq_u32(diff, two_p), diff)
    }
}

#[inline]
unsafe fn radix4_dif_neon(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    // SAFETY: the caller proves the four quarter spans and twiddle spans;
    // these constants are NEON registers selected by the AArch64 predicate.
    let (two_p, prime_2, mu_2, vec_p) = unsafe {
        (
            vdupq_n_u32(prime.wrapping_mul(2)),
            vdup_n_u32(prime),
            vdup_n_u32(neg_inverse),
            vdupq_n_u32(prime),
        )
    };
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);
    let mut index = 0_usize;
    while index.wrapping_add(4) <= quarter_len {
        // SAFETY: caller proves four value quarters and two twiddle quarters.
        let (a, b, c, d, tw0, tw1) = unsafe {
            (
                vld1q_u32(values.add(index)),
                vld1q_u32(values.add(quarter_len + index)),
                vld1q_u32(values.add(q2 + index)),
                vld1q_u32(values.add(q3 + index)),
                vld1q_u32(twiddles.add(index)),
                vld1q_u32(twiddles.add(quarter_len + index)),
            )
        };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let second_twiddle = unsafe { monty_mul_neon_4(tw0, tw0, vec_p, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_sum = unsafe { add_lazy_neon_4(a, c, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_diff = unsafe { sub_lazy_neon_4(a, c, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_sum = unsafe { add_lazy_neon_4(b, d, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_diff = unsafe { sub_lazy_neon_4(b, d, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_twiddled = unsafe { monty_mul_lazy_neon_4(low_diff, tw0, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_twiddled = unsafe { monty_mul_lazy_neon_4(high_diff, tw1, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out0 = unsafe { add_lazy_neon_4(low_sum, high_sum, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out1 = unsafe {
            monty_mul_lazy_neon_4(
                sub_lazy_neon_4(low_sum, high_sum, two_p),
                second_twiddle,
                mu_2,
                prime_2,
            )
        };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out2 = unsafe { add_lazy_neon_4(low_twiddled, high_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out3 = unsafe {
            monty_mul_lazy_neon_4(
                sub_lazy_neon_4(low_twiddled, high_twiddled, two_p),
                second_twiddle,
                mu_2,
                prime_2,
            )
        };
        // SAFETY: the same quarter-span proof covers all stores.
        unsafe {
            vst1q_u32(values.add(index), out0);
            vst1q_u32(values.add(quarter_len + index), out1);
            vst1q_u32(values.add(q2 + index), out2);
            vst1q_u32(values.add(q3 + index), out3);
        }
        index = index.wrapping_add(4);
    }
    while index < quarter_len {
        // SAFETY: scalar tail stays inside validated spans.
        unsafe { radix4_dif_one(values, twiddles, index, quarter_len, prime, neg_inverse) };
        index = index.wrapping_add(1);
    }
}

#[inline]
unsafe fn radix4_dit_neon(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    // SAFETY: the caller proves the four quarter spans and twiddle spans;
    // these constants are NEON registers selected by the AArch64 predicate.
    let (two_p, prime_2, mu_2, vec_p) = unsafe {
        (
            vdupq_n_u32(prime.wrapping_mul(2)),
            vdup_n_u32(prime),
            vdup_n_u32(neg_inverse),
            vdupq_n_u32(prime),
        )
    };
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);
    let mut index = 0_usize;
    while index.wrapping_add(4) <= quarter_len {
        // SAFETY: caller proves four value quarters and two twiddle quarters.
        let (a, b, c, d, tw0, tw1) = unsafe {
            (
                vld1q_u32(values.add(index)),
                vld1q_u32(values.add(quarter_len + index)),
                vld1q_u32(values.add(q2 + index)),
                vld1q_u32(values.add(q3 + index)),
                vld1q_u32(twiddles.add(index)),
                vld1q_u32(twiddles.add(quarter_len + index)),
            )
        };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let second_twiddle = unsafe { monty_mul_neon_4(tw0, tw0, vec_p, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_twiddled = unsafe { monty_mul_lazy_neon_4(b, second_twiddle, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_twiddled = unsafe { monty_mul_lazy_neon_4(d, second_twiddle, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_sum = unsafe { add_lazy_neon_4(a, low_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let low_diff = unsafe { sub_lazy_neon_4(a, low_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_sum = unsafe { add_lazy_neon_4(c, high_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_diff = unsafe { sub_lazy_neon_4(c, high_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_sum_twiddled = unsafe { monty_mul_lazy_neon_4(high_sum, tw0, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let high_diff_twiddled = unsafe { monty_mul_lazy_neon_4(high_diff, tw1, mu_2, prime_2) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out0 = unsafe { add_lazy_neon_4(low_sum, high_sum_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out1 = unsafe { add_lazy_neon_4(low_diff, high_diff_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out2 = unsafe { sub_lazy_neon_4(low_sum, high_sum_twiddled, two_p) };
        // SAFETY: NEON is selected by the AArch64 backend; all operands are initialized vectors.
        let out3 = unsafe { sub_lazy_neon_4(low_diff, high_diff_twiddled, two_p) };
        // SAFETY: the same quarter-span proof covers all stores.
        unsafe {
            vst1q_u32(values.add(index), out0);
            vst1q_u32(values.add(quarter_len + index), out1);
            vst1q_u32(values.add(q2 + index), out2);
            vst1q_u32(values.add(q3 + index), out3);
        }
        index = index.wrapping_add(4);
    }
    while index < quarter_len {
        // SAFETY: scalar tail stays inside validated spans.
        unsafe { radix4_dit_one(values, twiddles, index, quarter_len, prime, neg_inverse) };
        index = index.wrapping_add(1);
    }
}

#[inline]
pub fn ntt_monty_u32() -> NttMontyKernels {
    NttMontyKernels {
        mul_slice: monty_mul_slice_neon,
        dif_butterfly: dif_butterfly_neon,
        dit_butterfly: dit_butterfly_neon,
        radix4_dif: radix4_dif_neon,
        radix4_dit: radix4_dit_neon,
    }
}
