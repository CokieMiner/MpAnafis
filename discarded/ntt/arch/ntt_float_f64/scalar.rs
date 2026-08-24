//! Portable scalar fallback for 50-bit floating-point Harvey NTT butterflies.

#![allow(
    unsafe_code,
    clippy::suboptimal_flops,
    reason = "Target fallback for architectures without hardware FMA intrinsics"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Standard mathematical notation for FFT algorithms and Dekker TwoProduct"
)]

const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;
const DEKKER_SPLIT: f64 = 134_217_729.0;

#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let c = DEKKER_SPLIT * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    let d = DEKKER_SPLIT * b;
    let b_hi = d - (d - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
pub fn reduce_to_pm1n_scalar(a: f64, prime: f64, pinv: f64) -> f64 {
    let q = ((a * pinv) + ROUND_MAGIC) - ROUND_MAGIC;
    a - q * prime
}

#[inline]
pub fn mulmod_scalar(a: f64, b: f64, prime: f64, pinv: f64) -> f64 {
    let (h_ab, l_ab) = two_product(a, b);
    let q = ((h_ab * pinv) + ROUND_MAGIC) - ROUND_MAGIC;
    let (h_qp, l_qp) = two_product(q, prime);
    let diff_h = h_ab - h_qp;
    (diff_h - l_qp) + l_ab
}

/// Applies one fused radix-4 floating-point DIF butterfly group at `index`.
///
/// # Safety
/// `values` covers four disjoint quarter spans of length `quarter_len`, and
/// `twiddles` covers three spans of that length. All residues satisfy the 50-bit contract.
pub unsafe fn radix4_dif_float_one(
    values: *mut f64,
    twiddles: *const f64,
    index: usize,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);

    // SAFETY: the caller proves all quarter offsets and twiddle offsets.
    let (a_raw, b, c, d, tw0, tw1, second_twiddle) = unsafe {
        (
            *values.add(index),
            *values.add(quarter_len.wrapping_add(index)),
            *values.add(q2.wrapping_add(index)),
            *values.add(q3.wrapping_add(index)),
            *twiddles.add(index),
            *twiddles.add(quarter_len.wrapping_add(index)),
            *twiddles.add(q2.wrapping_add(index)),
        )
    };

    let a = reduce_to_pm1n_scalar(a_raw, prime, pinv);

    let low_sum = reduce_to_pm1n_scalar(a + c, prime, pinv);
    let low_diff = mulmod_scalar(a - c, tw0, prime, pinv);
    let high_sum = reduce_to_pm1n_scalar(b + d, prime, pinv);
    let high_diff = mulmod_scalar(b - d, tw1, prime, pinv);

    let out0 = low_sum + high_sum;
    let out1 = mulmod_scalar(low_sum - high_sum, second_twiddle, prime, pinv);
    let out2 = low_diff + high_diff;
    let out3 = mulmod_scalar(low_diff - high_diff, second_twiddle, prime, pinv);

    // SAFETY: the caller proves the four output slots are writable.
    unsafe {
        *values.add(index) = out0;
        *values.add(quarter_len.wrapping_add(index)) = out1;
        *values.add(q2.wrapping_add(index)) = out2;
        *values.add(q3.wrapping_add(index)) = out3;
    }
}

/// Applies one fused radix-4 floating-point DIT butterfly group at `index`.
///
/// # Safety
/// Same span and residue preconditions as [`radix4_dif_float_one`].
pub unsafe fn radix4_dit_float_one(
    values: *mut f64,
    twiddles: *const f64,
    index: usize,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    let q2 = quarter_len.wrapping_mul(2);
    let q3 = quarter_len.wrapping_mul(3);

    // SAFETY: the caller proves all quarter offsets and twiddle offsets.
    let (a, b, c, d, tw0, tw1, second_twiddle) = unsafe {
        (
            *values.add(index),
            *values.add(quarter_len.wrapping_add(index)),
            *values.add(q2.wrapping_add(index)),
            *values.add(q3.wrapping_add(index)),
            *twiddles.add(index),
            *twiddles.add(quarter_len.wrapping_add(index)),
            *twiddles.add(q2.wrapping_add(index)),
        )
    };

    let low_twiddled = mulmod_scalar(b, second_twiddle, prime, pinv);
    let high_twiddled = mulmod_scalar(d, second_twiddle, prime, pinv);

    let low_sum = reduce_to_pm1n_scalar(a + low_twiddled, prime, pinv);
    let low_diff = reduce_to_pm1n_scalar(a - low_twiddled, prime, pinv);
    let high_sum = reduce_to_pm1n_scalar(c + high_twiddled, prime, pinv);
    let high_diff = reduce_to_pm1n_scalar(c - high_twiddled, prime, pinv);

    let high_sum_twiddled = mulmod_scalar(high_sum, tw0, prime, pinv);
    let high_diff_twiddled = mulmod_scalar(high_diff, tw1, prime, pinv);

    let out0 = low_sum + high_sum_twiddled;
    let out1 = low_diff + high_diff_twiddled;
    let out2 = low_sum - high_sum_twiddled;
    let out3 = low_diff - high_diff_twiddled;

    // SAFETY: the caller proves the four output slots are writable.
    unsafe {
        *values.add(index) = out0;
        *values.add(quarter_len.wrapping_add(index)) = out1;
        *values.add(q2.wrapping_add(index)) = out2;
        *values.add(q3.wrapping_add(index)) = out3;
    }
}

/// Branchless scalar Radix-4 decimation-in-frequency step.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `2 * quarter_len` `f64` elements.
#[allow(
    dead_code,
    reason = "Used by fallback.rs and _internal-tune on architectures without native NEON"
)]
pub unsafe fn radix4_dif_float_scalar(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    let mut idx = 0_usize;
    while idx < quarter_len {
        // SAFETY: caller guarantees pointers and spans are valid for 4 * quarter_len.
        unsafe {
            radix4_dif_float_one(values, twiddles, idx, quarter_len, prime, pinv);
        }
        idx = idx.wrapping_add(1);
    }
}

/// Branchless scalar Radix-4 decimation-in-time step.
///
/// # Safety
/// - `values` is valid for reads and writes of `4 * quarter_len` `f64` elements.
/// - `twiddles` is valid for reads of `2 * quarter_len` `f64` elements.
#[allow(
    dead_code,
    reason = "Used by fallback.rs and _internal-tune on architectures without native NEON"
)]
pub unsafe fn radix4_dit_float_scalar(
    values: *mut f64,
    twiddles: *const f64,
    quarter_len: usize,
    prime: f64,
    pinv: f64,
) {
    let mut idx = 0_usize;
    while idx < quarter_len {
        // SAFETY: caller guarantees pointers and spans are valid for 4 * quarter_len.
        unsafe {
            radix4_dit_float_one(values, twiddles, idx, quarter_len, prime, pinv);
        }
        idx = idx.wrapping_add(1);
    }
}

/// Pointwise frequency-domain multiplication of slices `a` and `b`.
///
/// # Safety
/// `a` and `b` are valid for reading `len` elements, and `a` is valid for writing `len` elements.
#[allow(
    dead_code,
    reason = "Used by fallback.rs and _internal-tune on architectures without native NEON"
)]
pub unsafe fn pointwise_mul_float_scalar(
    a: *mut f64,
    b: *const f64,
    len: usize,
    prime: f64,
    pinv: f64,
) {
    let mut i = 0_usize;
    while i < len {
        // SAFETY: caller establishes len elements for a and b.
        unsafe {
            let va = *a.add(i);
            let vb = *b.add(i);
            *a.add(i) = mulmod_scalar(va, vb, prime, pinv);
        }
        i = i.wrapping_add(1);
    }
}

/// Pointwise frequency-domain squaring of slice `a`.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[allow(
    dead_code,
    reason = "Used by fallback.rs and _internal-tune on architectures without native NEON"
)]
pub unsafe fn pointwise_sqr_float_scalar(a: *mut f64, len: usize, prime: f64, pinv: f64) {
    let mut i = 0_usize;
    while i < len {
        // SAFETY: caller establishes len elements for a.
        unsafe {
            let va = *a.add(i);
            *a.add(i) = mulmod_scalar(va, va, prime, pinv);
        }
        i = i.wrapping_add(1);
    }
}

/// Scales slice `a` by scalar `inv_n` modulo `prime`.
///
/// # Safety
/// `a` is valid for reading and writing `len` elements.
#[allow(
    dead_code,
    reason = "Used by fallback.rs and _internal-tune on architectures without native NEON"
)]
pub unsafe fn scale_float_scalar(a: *mut f64, len: usize, inv_n: f64, prime: f64, pinv: f64) {
    let mut i = 0_usize;
    while i < len {
        // SAFETY: caller establishes len elements for a.
        unsafe {
            let va = *a.add(i);
            *a.add(i) = mulmod_scalar(va, inv_n, prime, pinv);
        }
        i = i.wrapping_add(1);
    }
}
