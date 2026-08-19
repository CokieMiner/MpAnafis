//! AVX2-accelerated 8-lane Montgomery kernels for 31-bit moduli with Harvey lazy reduction.

#![allow(
    unsafe_code,
    reason = "Target feature AVX2 intrinsics natively require unsafe code"
)]
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::arithmetic_side_effects,
    clippy::semicolon_inside_block,
    reason = "Vector broadcast and modular casts are intentional on 32-bit values"
)]
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_blend_epi32, _mm256_blendv_epi8,
    _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_min_epu32, _mm256_mul_epu32, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_slli_si256, _mm256_srli_epi64, _mm256_srli_si256,
    _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_epi64,
};

use super::{NttMontyKernels, radix4_dif_one, radix4_dit_one};

#[target_feature(enable = "avx2")]
unsafe fn monty_mul_avx2_8(a: __m256i, b: __m256i, vec_p: __m256i, vec_mu: __m256i) -> __m256i {
    let t_even = _mm256_mul_epu32(a, b);
    let q_even = _mm256_mul_epu32(t_even, vec_mu);
    let q_p_even = _mm256_mul_epu32(q_even, vec_p);
    let t_plus_qp_even = _mm256_add_epi64(t_even, q_p_even);
    let m_even = _mm256_srli_epi64(t_plus_qp_even, 32);
    let m_sub_p_even = _mm256_sub_epi64(m_even, vec_p);
    let red_even = _mm256_min_epu32(m_even, m_sub_p_even);

    let a_odd = _mm256_srli_si256(a, 4);
    let b_odd = _mm256_srli_si256(b, 4);
    let t_odd = _mm256_mul_epu32(a_odd, b_odd);
    let q_odd = _mm256_mul_epu32(t_odd, vec_mu);
    let q_p_odd = _mm256_mul_epu32(q_odd, vec_p);
    let t_plus_qp_odd = _mm256_add_epi64(t_odd, q_p_odd);
    let m_odd = _mm256_srli_epi64(t_plus_qp_odd, 32);
    let m_sub_p_odd = _mm256_sub_epi64(m_odd, vec_p);
    let red_odd = _mm256_min_epu32(m_odd, m_sub_p_odd);

    let shifted_odd = _mm256_slli_si256(red_odd, 4);
    _mm256_blend_epi32(red_even, shifted_odd, 0xAA)
}

/// Harvey lazy Montgomery multiplication: given a in [0, 2p) and b in [0, p),
/// computes (a * b * R^-1) mod p in [0, 2p) with zero conditional subtractions.
#[target_feature(enable = "avx2")]
unsafe fn monty_mul_lazy_avx2_8(
    a: __m256i,
    b: __m256i,
    vec_p: __m256i,
    vec_mu: __m256i,
) -> __m256i {
    let t_even = _mm256_mul_epu32(a, b);
    let q_even = _mm256_mul_epu32(t_even, vec_mu);
    let q_p_even = _mm256_mul_epu32(q_even, vec_p);
    let t_plus_qp_even = _mm256_add_epi64(t_even, q_p_even);
    let m_even = _mm256_srli_epi64(t_plus_qp_even, 32);

    let a_odd = _mm256_srli_si256(a, 4);
    let b_odd = _mm256_srli_si256(b, 4);
    let t_odd = _mm256_mul_epu32(a_odd, b_odd);
    let q_odd = _mm256_mul_epu32(t_odd, vec_mu);
    let q_p_odd = _mm256_mul_epu32(q_odd, vec_p);
    let t_plus_qp_odd = _mm256_add_epi64(t_odd, q_p_odd);
    let m_odd = _mm256_srli_epi64(t_plus_qp_odd, 32);

    let shifted_odd = _mm256_slli_si256(m_odd, 4);
    _mm256_blend_epi32(m_even, shifted_odd, 0xAA)
}

#[target_feature(enable = "avx2")]
unsafe fn add_lazy_avx2(a: __m256i, b: __m256i, two_p: __m256i, wrap: __m256i) -> __m256i {
    let sum = _mm256_add_epi32(a, b);
    let reduced = _mm256_min_epu32(sum, _mm256_sub_epi32(sum, two_p));
    let no_carry = _mm256_cmpeq_epi32(_mm256_min_epu32(sum, a), a);
    _mm256_blendv_epi8(_mm256_add_epi32(sum, wrap), reduced, no_carry)
}

#[target_feature(enable = "avx2")]
unsafe fn sub_lazy_avx2(a: __m256i, b: __m256i, two_p: __m256i) -> __m256i {
    let diff = _mm256_sub_epi32(a, b);
    let no_borrow = _mm256_cmpeq_epi32(_mm256_min_epu32(a, b), b);
    _mm256_blendv_epi8(_mm256_add_epi32(diff, two_p), diff, no_borrow)
}

#[target_feature(enable = "avx2")]
pub unsafe fn radix4_dif_avx2(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let vec_p = _mm256_set1_epi64x(i64::from(prime));
    let vec_mu = _mm256_set1_epi64x(i64::from(neg_inverse));
    let two_p = prime.wrapping_mul(2);
    let vec_2p = _mm256_set1_epi32(two_p as i32);
    let vec_wrap = _mm256_set1_epi32(0_u32.wrapping_sub(two_p) as i32);
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);
    let mut index = 0_usize;
    while index.wrapping_add(8) <= quarter_len {
        // SAFETY: caller proves four value quarters and two twiddle quarters.
        let (a, b, c, d, tw0, tw1) = unsafe {
            (
                _mm256_loadu_si256(values.add(index).cast()),
                _mm256_loadu_si256(values.add(quarter_len + index).cast()),
                _mm256_loadu_si256(values.add(q2 + index).cast()),
                _mm256_loadu_si256(values.add(q3 + index).cast()),
                _mm256_loadu_si256(twiddles.add(index).cast()),
                _mm256_loadu_si256(twiddles.add(quarter_len + index).cast()),
            )
        };
        // SAFETY: AVX2 target-feature is enabled; all values are initialized
        // vectors and the public span contract validates the loaded lanes.
        let second_twiddle = unsafe { monty_mul_avx2_8(tw0, tw0, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let low_sum = unsafe { add_lazy_avx2(a, c, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let low_diff = unsafe { sub_lazy_avx2(a, c, vec_2p) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let high_sum = unsafe { add_lazy_avx2(b, d, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let high_diff = unsafe { sub_lazy_avx2(b, d, vec_2p) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let low_twiddled = unsafe { monty_mul_lazy_avx2_8(low_diff, tw0, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let high_twiddled = unsafe { monty_mul_lazy_avx2_8(high_diff, tw1, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let out0 = unsafe { add_lazy_avx2(low_sum, high_sum, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let out1 = unsafe {
            monty_mul_lazy_avx2_8(
                sub_lazy_avx2(low_sum, high_sum, vec_2p),
                second_twiddle,
                vec_p,
                vec_mu,
            )
        };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let out2 = unsafe { add_lazy_avx2(low_twiddled, high_twiddled, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and the vector operands are initialized.
        let out3 = unsafe {
            monty_mul_lazy_avx2_8(
                sub_lazy_avx2(low_twiddled, high_twiddled, vec_2p),
                second_twiddle,
                vec_p,
                vec_mu,
            )
        };
        // SAFETY: the same quarter-span proof covers all stores.
        unsafe {
            _mm256_storeu_si256(values.add(index).cast(), out0);
            _mm256_storeu_si256(values.add(quarter_len + index).cast(), out1);
            _mm256_storeu_si256(values.add(q2 + index).cast(), out2);
            _mm256_storeu_si256(values.add(q3 + index).cast(), out3);
        }
        index = index.wrapping_add(8);
    }
    while index < quarter_len {
        // SAFETY: the scalar tail remains inside the same validated spans.
        unsafe { radix4_dif_one(values, twiddles, index, quarter_len, prime, neg_inverse) };
        index = index.wrapping_add(1);
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn radix4_dit_avx2(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let vec_p = _mm256_set1_epi64x(i64::from(prime));
    let vec_mu = _mm256_set1_epi64x(i64::from(neg_inverse));
    let two_p = prime.wrapping_mul(2);
    let vec_2p = _mm256_set1_epi32(two_p as i32);
    let vec_wrap = _mm256_set1_epi32(0_u32.wrapping_sub(two_p) as i32);
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);
    let mut index = 0_usize;
    while index.wrapping_add(8) <= quarter_len {
        // SAFETY: caller proves four value quarters and two twiddle quarters.
        let (a, b, c, d, tw0, tw1) = unsafe {
            (
                _mm256_loadu_si256(values.add(index).cast()),
                _mm256_loadu_si256(values.add(quarter_len + index).cast()),
                _mm256_loadu_si256(values.add(q2 + index).cast()),
                _mm256_loadu_si256(values.add(q3 + index).cast()),
                _mm256_loadu_si256(twiddles.add(index).cast()),
                _mm256_loadu_si256(twiddles.add(quarter_len + index).cast()),
            )
        };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let second_twiddle = unsafe { monty_mul_avx2_8(tw0, tw0, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let low_twiddled = unsafe { monty_mul_lazy_avx2_8(b, second_twiddle, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let high_twiddled = unsafe { monty_mul_lazy_avx2_8(d, second_twiddle, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let low_sum = unsafe { add_lazy_avx2(a, low_twiddled, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let low_diff = unsafe { sub_lazy_avx2(a, low_twiddled, vec_2p) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let high_sum = unsafe { add_lazy_avx2(c, high_twiddled, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let high_diff = unsafe { sub_lazy_avx2(c, high_twiddled, vec_2p) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let high_sum_twiddled = unsafe { monty_mul_lazy_avx2_8(high_sum, tw0, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let high_diff_twiddled = unsafe { monty_mul_lazy_avx2_8(high_diff, tw1, vec_p, vec_mu) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let out0 = unsafe { add_lazy_avx2(low_sum, high_sum_twiddled, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let out1 = unsafe { add_lazy_avx2(low_diff, high_diff_twiddled, vec_2p, vec_wrap) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let out2 = unsafe { sub_lazy_avx2(low_sum, high_sum_twiddled, vec_2p) };
        // SAFETY: AVX2 target-feature is enabled and all vector operands are initialized.
        let out3 = unsafe { sub_lazy_avx2(low_diff, high_diff_twiddled, vec_2p) };
        // SAFETY: the same quarter-span proof covers all stores.
        unsafe {
            _mm256_storeu_si256(values.add(index).cast(), out0);
            _mm256_storeu_si256(values.add(quarter_len + index).cast(), out1);
            _mm256_storeu_si256(values.add(q2 + index).cast(), out2);
            _mm256_storeu_si256(values.add(q3 + index).cast(), out3);
        }
        index = index.wrapping_add(8);
    }
    while index < quarter_len {
        // SAFETY: the scalar tail remains inside the same validated spans.
        unsafe { radix4_dit_one(values, twiddles, index, quarter_len, prime, neg_inverse) };
        index = index.wrapping_add(1);
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn monty_mul_slice_avx2(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let vec_p_64 = _mm256_set1_epi64x(i64::from(prime));
    let vec_mu_64 = _mm256_set1_epi64x(i64::from(neg_inverse));
    let vec_p_32 = _mm256_set1_epi32(prime as i32);

    let mut i = 0_usize;
    while i.wrapping_add(8) <= len {
        // SAFETY: caller guarantees both input spans are readable for length len.
        let (va, vb) = unsafe {
            (
                _mm256_loadu_si256(a.add(i).cast()),
                _mm256_loadu_si256(b.add(i).cast()),
            )
        };
        let va_normalized = _mm256_min_epu32(va, _mm256_sub_epi32(va, vec_p_32));
        let vb_normalized = _mm256_min_epu32(vb, _mm256_sub_epi32(vb, vec_p_32));

        // SAFETY: AVX2 enabled in this target_feature fn.
        let vres = unsafe { monty_mul_avx2_8(va_normalized, vb_normalized, vec_p_64, vec_mu_64) };
        // SAFETY: caller guarantees dst writable for length len.
        unsafe {
            _mm256_storeu_si256(dst.add(i).cast(), vres);
        }
        i = i.wrapping_add(8);
    }
    while i < len {
        // SAFETY: caller guarantees both input spans are readable for length len.
        let (mut x, mut y) = unsafe { (*a.add(i), *b.add(i)) };
        if x >= prime {
            x -= prime;
        }
        if y >= prime {
            y -= prime;
        }
        let t = u64::from(x).wrapping_mul(u64::from(y));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let m =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;
        let res = if m >= prime { m.wrapping_sub(prime) } else { m };
        // SAFETY: caller guarantees dst writable for length len.
        unsafe {
            *dst.add(i) = res;
        }
        i = i.wrapping_add(1);
    }
}

/// Harvey lazy DIF butterfly:
/// u' = (u + v) mod 2p in [0, 2p)
/// v' = ((u - v) mod 2p * tw) mod p in [0, 2p)
#[target_feature(enable = "avx2")]
pub unsafe fn dif_butterfly_avx2(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let vec_p_64 = _mm256_set1_epi64x(i64::from(prime));
    let vec_mu_64 = _mm256_set1_epi64x(i64::from(neg_inverse));
    let two_p = prime.wrapping_mul(2);
    let vec_2p_32 = _mm256_set1_epi32(two_p as i32);

    let mut i = 0_usize;
    while i.wrapping_add(8) <= len {
        // SAFETY: caller guarantees all three spans are readable for length len.
        let (u, v, tw) = unsafe {
            (
                _mm256_loadu_si256(low.add(i).cast()),
                _mm256_loadu_si256(high.add(i).cast()),
                _mm256_loadu_si256(twiddles.add(i).cast()),
            )
        };

        let vec_wrap = _mm256_set1_epi32(0_u32.wrapping_sub(two_p) as i32);
        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let new_u = unsafe { add_lazy_avx2(u, v, vec_2p_32, vec_wrap) };
        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let diff_mod_2p = unsafe { sub_lazy_avx2(u, v, vec_2p_32) };

        // v' = (diff * tw) in [0, 2p)
        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let new_v = unsafe { monty_mul_lazy_avx2_8(diff_mod_2p, tw, vec_p_64, vec_mu_64) };

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            _mm256_storeu_si256(low.add(i).cast(), new_u);
            _mm256_storeu_si256(high.add(i).cast(), new_v);
        }
        i = i.wrapping_add(8);
    }
    while i < len {
        // SAFETY: caller guarantees all three spans are readable at index i.
        let (u, v, tw) = unsafe { (*low.add(i), *high.add(i), *twiddles.add(i)) };

        let sum = u64::from(u).wrapping_add(u64::from(v));
        // Both branches are below 2p < 2^32, so these narrowing casts are exact.
        let new_u = if sum >= u64::from(two_p) {
            sum.wrapping_sub(u64::from(two_p)) as u32
        } else {
            sum as u32
        };
        let diff = if u >= v {
            u.wrapping_sub(v)
        } else {
            (u64::from(u)
                .wrapping_add(u64::from(two_p))
                .wrapping_sub(u64::from(v))) as u32
        };

        let t = u64::from(diff).wrapping_mul(u64::from(tw));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let new_v =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            *low.add(i) = new_u;
            *high.add(i) = new_v;
        }
        i = i.wrapping_add(1);
    }
}

/// Harvey lazy DIT butterfly:
/// prod = (v * tw) mod p in [0, 2p)
/// u' = (u + prod) mod 2p in [0, 2p)
/// v' = (u - prod) mod 2p in [0, 2p)
#[target_feature(enable = "avx2")]
pub unsafe fn dit_butterfly_avx2(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let vec_p_64 = _mm256_set1_epi64x(i64::from(prime));
    let vec_mu_64 = _mm256_set1_epi64x(i64::from(neg_inverse));
    let two_p = prime.wrapping_mul(2);
    let vec_2p_32 = _mm256_set1_epi32(two_p as i32);
    let vec_wrap = _mm256_set1_epi32(0_u32.wrapping_sub(two_p) as i32);

    let mut i = 0_usize;
    while i.wrapping_add(8) <= len {
        // SAFETY: caller guarantees all three spans are readable for length len.
        let (u, v, tw) = unsafe {
            (
                _mm256_loadu_si256(low.add(i).cast()),
                _mm256_loadu_si256(high.add(i).cast()),
                _mm256_loadu_si256(twiddles.add(i).cast()),
            )
        };

        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let prod = unsafe { monty_mul_lazy_avx2_8(v, tw, vec_p_64, vec_mu_64) };
        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let new_u = unsafe { add_lazy_avx2(u, prod, vec_2p_32, vec_wrap) };
        // SAFETY: this helper and its caller both execute with AVX2 enabled.
        let new_v = unsafe { sub_lazy_avx2(u, prod, vec_2p_32) };

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            _mm256_storeu_si256(low.add(i).cast(), new_u);
            _mm256_storeu_si256(high.add(i).cast(), new_v);
        }
        i = i.wrapping_add(8);
    }
    while i < len {
        // SAFETY: caller guarantees all three spans are readable at index i.
        let (u, v, tw) = unsafe { (*low.add(i), *high.add(i), *twiddles.add(i)) };

        let t = u64::from(v).wrapping_mul(u64::from(tw));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let prod =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;

        let sum = u64::from(u).wrapping_add(u64::from(prod));
        // Both branches are below 2p < 2^32, so these narrowing casts are exact.
        let new_u = if sum >= u64::from(two_p) {
            sum.wrapping_sub(u64::from(two_p)) as u32
        } else {
            sum as u32
        };
        let diff = if u >= prod {
            u.wrapping_sub(prod)
        } else {
            (u64::from(u)
                .wrapping_add(u64::from(two_p))
                .wrapping_sub(u64::from(prod))) as u32
        };

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            *low.add(i) = new_u;
            *high.add(i) = diff;
        }
        i = i.wrapping_add(1);
    }
}

#[inline]
pub fn ntt_monty_u32() -> NttMontyKernels {
    NttMontyKernels {
        mul_slice: monty_mul_slice_avx2,
        dif_butterfly: dif_butterfly_avx2,
        dit_butterfly: dit_butterfly_avx2,
        radix4_dif: radix4_dif_avx2,
        radix4_dit: radix4_dit_avx2,
    }
}
