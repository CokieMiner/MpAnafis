//! Baseline x86-64 31-bit Montgomery NTT kernels with Harvey lazy butterflies.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "Montgomery REDC extracts low radix word; REDC proves reduction < 2p < 2^32"
)]

use super::{NttMontyKernels, radix4_dif_scalar, radix4_dit_scalar};

#[inline]
pub unsafe fn monty_mul_slice_sse2(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for i in 0..len {
        // SAFETY: caller guarantees valid spans of length len.
        let mut x = unsafe { *a.add(i) };
        // SAFETY: caller guarantees valid spans of length len.
        let mut y = unsafe { *b.add(i) };
        if x >= prime {
            x = x.wrapping_sub(prime);
        }
        if y >= prime {
            y = y.wrapping_sub(prime);
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
    }
}

#[inline]
pub unsafe fn dif_butterfly_sse2(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for i in 0..len {
        // SAFETY: caller guarantees low, high, and twiddles valid for length len.
        let u = unsafe { *low.add(i) };
        // SAFETY: caller guarantees valid spans.
        let v = unsafe { *high.add(i) };
        // SAFETY: caller guarantees valid spans.
        let w = unsafe { *twiddles.add(i) };

        // Lazy operands may both be as large as 2p.  Their sum needs 33 bits;
        // widen before reducing modulo 2p instead of wrapping a u32 sum.
        let two_p = u64::from(prime) * 2;
        let sum = u64::from(u).wrapping_add(u64::from(v));
        // Both branches are below 2p < 2^32, so these narrowing casts are exact.
        let new_u = if sum >= two_p {
            sum.wrapping_sub(two_p) as u32
        } else {
            sum as u32
        };
        let diff = if u >= v {
            u.wrapping_sub(v)
        } else {
            (u64::from(u).wrapping_add(two_p).wrapping_sub(u64::from(v))) as u32
        };

        let t = u64::from(diff).wrapping_mul(u64::from(w));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let new_v =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            *low.add(i) = new_u;
            *high.add(i) = new_v;
        }
    }
}

#[inline]
pub unsafe fn dit_butterfly_sse2(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for i in 0..len {
        // SAFETY: caller guarantees low, high, and twiddles valid for length len.
        let u = unsafe { *low.add(i) };
        // SAFETY: caller guarantees valid spans.
        let v = unsafe { *high.add(i) };
        // SAFETY: caller guarantees valid spans.
        let w = unsafe { *twiddles.add(i) };

        let t = u64::from(v).wrapping_mul(u64::from(w));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let m =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;
        let prod = if m >= prime { m.wrapping_sub(prime) } else { m };

        // As in DIF, use widened arithmetic for the lazy [0, 2p) add.
        let two_p = u64::from(prime) * 2;
        let sum = u64::from(u).wrapping_add(u64::from(prod));
        // Both branches are below 2p < 2^32, so these narrowing casts are exact.
        let new_u = if sum >= two_p {
            sum.wrapping_sub(two_p) as u32
        } else {
            sum as u32
        };
        let new_v = if u >= prod {
            u.wrapping_sub(prod)
        } else {
            (u64::from(u)
                .wrapping_add(two_p)
                .wrapping_sub(u64::from(prod))) as u32
        };

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            *low.add(i) = new_u;
            *high.add(i) = new_v;
        }
    }
}

#[inline]
pub fn ntt_monty_u32() -> NttMontyKernels {
    NttMontyKernels {
        mul_slice: monty_mul_slice_sse2,
        dif_butterfly: dif_butterfly_sse2,
        dit_butterfly: dit_butterfly_sse2,
        radix4_dif: radix4_dif_scalar,
        radix4_dit: radix4_dit_scalar,
    }
}
