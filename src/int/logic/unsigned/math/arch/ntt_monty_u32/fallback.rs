//! Portable fallback for 31-bit Montgomery NTT kernels with Harvey lazy reduction.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "Montgomery REDC extracts low radix word; REDC proves reduction < 2p < 2^32"
)]

use super::{NttMontyKernels, radix4_dif_scalar, radix4_dit_scalar};

#[inline]
pub unsafe fn monty_mul_slice_fallback(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for i in 0..len {
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
    }
}

#[inline]
pub unsafe fn dif_butterfly_fallback(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let two_p = prime.wrapping_mul(2);
    for i in 0..len {
        // SAFETY: caller guarantees low, high, and twiddles readable for length len.
        let (u, v, w) = unsafe { (*low.add(i), *high.add(i), *twiddles.add(i)) };

        // The lazy inputs are each below 2p, so their sum needs 33 bits.  Keep
        // this scalar reference widened; wrapping before comparing with 2p is
        // wrong for the configured prime near 2^31.
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
pub unsafe fn dit_butterfly_fallback(
    low: *mut u32,
    high: *mut u32,
    twiddles: *const u32,
    len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let two_p = prime.wrapping_mul(2);
    for i in 0..len {
        // SAFETY: caller guarantees low, high, and twiddles readable for length len.
        let (u, v, w) = unsafe { (*low.add(i), *high.add(i), *twiddles.add(i)) };

        let t = u64::from(v).wrapping_mul(u64::from(w));
        let q = t.wrapping_mul(u64::from(neg_inverse)) as u32;
        let prod =
            (t.wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))).wrapping_shr(32) as u32;

        // `u` and `prod` are lazy residues.  Widen the add so the [0, 2p)
        // reduction remains correct when their 32-bit sum overflows.
        let sum = u64::from(u) + u64::from(prod);
        // Both branches are below 2p < 2^32, so these narrowing casts are exact.
        let new_u = if sum >= u64::from(two_p) {
            (sum - u64::from(two_p)) as u32
        } else {
            sum as u32
        };
        let diff = if u >= prod {
            u.wrapping_sub(prod)
        } else {
            (u64::from(u) + u64::from(two_p) - u64::from(prod)) as u32
        };

        // SAFETY: caller guarantees low and high writable for length len.
        unsafe {
            *low.add(i) = new_u;
            *high.add(i) = diff;
        }
    }
}

#[inline]
pub fn ntt_monty_u32() -> NttMontyKernels {
    NttMontyKernels {
        mul_slice: monty_mul_slice_fallback,
        dif_butterfly: dif_butterfly_fallback,
        dit_butterfly: dit_butterfly_fallback,
        radix4_dif: radix4_dif_scalar,
        radix4_dit: radix4_dit_scalar,
    }
}
