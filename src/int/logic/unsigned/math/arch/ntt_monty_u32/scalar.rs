//! Scalar fused radix-4 DIF/DIT reference kernels.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Montgomery REDC extracts a bounded 32-bit residue"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "Validated lazy residues keep widened butterfly sums and corrections within u64"
)]

#[inline]
fn monty_mul(a: u32, b: u32, prime: u32, neg_inverse: u32) -> u32 {
    let product = u64::from(a).wrapping_mul(u64::from(b));
    let q = product.wrapping_mul(u64::from(neg_inverse)) as u32;
    product
        .wrapping_add(u64::from(q).wrapping_mul(u64::from(prime)))
        .wrapping_shr(32) as u32
}

#[inline]
fn monty_mul_canonical(a: u32, b: u32, prime: u32, neg_inverse: u32) -> u32 {
    let reduced = monty_mul(a, b, prime, neg_inverse);
    if reduced >= prime {
        reduced - prime
    } else {
        reduced
    }
}

#[inline]
fn add_lazy(a: u32, b: u32, two_prime: u64) -> u32 {
    let sum = u64::from(a) + u64::from(b);
    if sum >= two_prime {
        (sum - two_prime) as u32
    } else {
        sum as u32
    }
}

#[inline]
fn sub_lazy(a: u32, b: u32, two_prime: u64) -> u32 {
    if a >= b {
        a.wrapping_sub(b)
    } else {
        (u64::from(a) + two_prime - u64::from(b)) as u32
    }
}

/// Applies one fused radix-4 DIF butterfly group at `index`.
///
/// # Safety
/// `values` covers four disjoint quarter spans of length `quarter_len`, and
/// `twiddles` covers two spans of that length. Inputs are lazy residues and
/// twiddles are canonical Montgomery residues.
pub unsafe fn radix4_dif_one(
    values: *mut u32,
    twiddles: *const u32,
    index: usize,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let two_prime = u64::from(prime) * 2;
    // SAFETY: the caller proves all quarter offsets and twiddle offsets.
    let (a, b, c, d, tw0, tw1) = unsafe {
        (
            *values.add(index),
            *values.add(quarter_len.wrapping_add(index)),
            *values.add(quarter_len.wrapping_mul(2).wrapping_add(index)),
            *values.add(quarter_len.wrapping_mul(3).wrapping_add(index)),
            *twiddles.add(index),
            *twiddles.add(quarter_len.wrapping_add(index)),
        )
    };
    let second_twiddle = monty_mul_canonical(tw0, tw0, prime, neg_inverse);
    let low_sum = add_lazy(a, c, two_prime);
    let low_diff = sub_lazy(a, c, two_prime);
    let high_sum = add_lazy(b, d, two_prime);
    let high_diff = sub_lazy(b, d, two_prime);
    let low_twiddled = monty_mul(low_diff, tw0, prime, neg_inverse);
    let high_twiddled = monty_mul(high_diff, tw1, prime, neg_inverse);
    let out0 = add_lazy(low_sum, high_sum, two_prime);
    let out1 = monty_mul(
        sub_lazy(low_sum, high_sum, two_prime),
        second_twiddle,
        prime,
        neg_inverse,
    );
    let out2 = add_lazy(low_twiddled, high_twiddled, two_prime);
    let out3 = monty_mul(
        sub_lazy(low_twiddled, high_twiddled, two_prime),
        second_twiddle,
        prime,
        neg_inverse,
    );
    // SAFETY: the caller proves the four output slots are writable.
    unsafe {
        *values.add(index) = out0;
        *values.add(quarter_len.wrapping_add(index)) = out1;
        *values.add(quarter_len.wrapping_mul(2).wrapping_add(index)) = out2;
        *values.add(quarter_len.wrapping_mul(3).wrapping_add(index)) = out3;
    }
}

/// Applies one fused radix-4 DIT butterfly group at `index`.
///
/// # Safety
/// Same span and residue preconditions as [`radix4_dif_one`].
pub unsafe fn radix4_dit_one(
    values: *mut u32,
    twiddles: *const u32,
    index: usize,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    let two_prime = u64::from(prime) * 2;
    // SAFETY: the caller proves all quarter offsets and twiddle offsets.
    let (a, b, c, d, tw0, tw1) = unsafe {
        (
            *values.add(index),
            *values.add(quarter_len.wrapping_add(index)),
            *values.add(quarter_len.wrapping_mul(2).wrapping_add(index)),
            *values.add(quarter_len.wrapping_mul(3).wrapping_add(index)),
            *twiddles.add(index),
            *twiddles.add(quarter_len.wrapping_add(index)),
        )
    };
    let second_twiddle = monty_mul_canonical(tw0, tw0, prime, neg_inverse);
    // DIT consumes the smaller (2q) butterflies first.  The two resulting
    // sums/differences are then combined by the 4q butterfly twiddles; doing
    // these operations in the opposite order changes the twiddle exponents.
    let low_twiddled = monty_mul(b, second_twiddle, prime, neg_inverse);
    let high_twiddled = monty_mul(d, second_twiddle, prime, neg_inverse);
    let low_sum = add_lazy(a, low_twiddled, two_prime);
    let low_diff = sub_lazy(a, low_twiddled, two_prime);
    let high_sum = add_lazy(c, high_twiddled, two_prime);
    let high_diff = sub_lazy(c, high_twiddled, two_prime);
    let high_sum_twiddled = monty_mul(high_sum, tw0, prime, neg_inverse);
    let high_diff_twiddled = monty_mul(high_diff, tw1, prime, neg_inverse);
    // SAFETY: the caller proves the four output slots are writable.
    unsafe {
        *values.add(index) = add_lazy(low_sum, high_sum_twiddled, two_prime);
        *values.add(quarter_len.wrapping_add(index)) =
            add_lazy(low_diff, high_diff_twiddled, two_prime);
        *values.add(quarter_len.wrapping_mul(2).wrapping_add(index)) =
            sub_lazy(low_sum, high_sum_twiddled, two_prime);
        *values.add(quarter_len.wrapping_mul(3).wrapping_add(index)) =
            sub_lazy(low_diff, high_diff_twiddled, two_prime);
    }
}

/// Applies fused radix-4 DIF groups across a complete four-quarter block.
#[cfg(any(
    test,
    feature = "_internal-tune",
    all(
        not(all(target_arch = "aarch64", target_pointer_width = "64")),
        not(all(target_arch = "x86_64", target_feature = "avx2"))
    )
))]
pub unsafe fn radix4_dif_scalar(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for index in 0..quarter_len {
        // SAFETY: the complete block spans and twiddle spans are caller-proven.
        unsafe {
            radix4_dif_one(values, twiddles, index, quarter_len, prime, neg_inverse);
        }
    }
}

/// Applies fused radix-4 DIT groups across a complete four-quarter block.
#[cfg(any(
    test,
    feature = "_internal-tune",
    all(
        not(all(target_arch = "aarch64", target_pointer_width = "64")),
        not(all(target_arch = "x86_64", target_feature = "avx2"))
    )
))]
pub unsafe fn radix4_dit_scalar(
    values: *mut u32,
    twiddles: *const u32,
    quarter_len: usize,
    prime: u32,
    neg_inverse: u32,
) {
    for index in 0..quarter_len {
        // SAFETY: the complete block spans and twiddle spans are caller-proven.
        unsafe {
            radix4_dit_one(values, twiddles, index, quarter_len, prime, neg_inverse);
        }
    }
}
