//! Portable fixed-width kernels for complete basecase multiplication.

use super::{DoubleLimb, Limb};

/// Write the exact product of two two-limb operands.
///
/// # Safety
///
/// `a` and `b` must each be readable for two limbs, `dst` must be writable
/// for four limbs, and neither input span may overlap `dst`.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "The four products are widened to DoubleLimb and fixed-column extraction is the measured two-limb hot path"
)]
#[inline(always)]
pub unsafe fn mul_2x2_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // SAFETY: the caller provides both exact two-limb input spans.
    let a0 = unsafe { *a } as DoubleLimb;
    // SAFETY: index one lies inside the caller-proven input span.
    let a1 = unsafe { *a.add(1) } as DoubleLimb;
    // SAFETY: the caller provides both exact two-limb input spans.
    let b0 = unsafe { *b } as DoubleLimb;
    // SAFETY: index one lies inside the caller-proven input span.
    let b1 = unsafe { *b.add(1) } as DoubleLimb;

    let product00 = a0.wrapping_mul(b0);
    let product01 = a0.wrapping_mul(b1);
    let product10 = a1.wrapping_mul(b0);
    let product11 = a1.wrapping_mul(b1);

    // Column one is bounded by 3B-4 and column two by 3B-3, far below
    // DoubleLimb's B^2 range. Their carries are therefore at most two.
    let column1 = product00
        .wrapping_shr(Limb::BITS)
        .wrapping_add(product01 as Limb as DoubleLimb)
        .wrapping_add(product10 as Limb as DoubleLimb);
    let column2 = product01
        .wrapping_shr(Limb::BITS)
        .wrapping_add(product10.wrapping_shr(Limb::BITS))
        .wrapping_add(product11 as Limb as DoubleLimb)
        .wrapping_add(column1.wrapping_shr(Limb::BITS));
    let top = product11
        .wrapping_shr(Limb::BITS)
        .wrapping_add(column2.wrapping_shr(Limb::BITS));
    debug_assert!(
        top <= Limb::MAX as DoubleLimb,
        "two-limb product overflowed four limbs"
    );

    // SAFETY: the caller provides exactly four writable output limbs. Each
    // cast extracts the corresponding radix-B digit of the exact product.
    unsafe {
        *dst = product00 as Limb;
        *dst.add(1) = column1 as Limb;
        *dst.add(2) = column2 as Limb;
        *dst.add(3) = top as Limb;
    }
}

/// Write the exact product of two three-limb operands with fixed loop bounds.
///
/// # Safety
///
/// `a` and `b` must each be readable for three limbs, `dst` must be writable
/// for six limbs, and neither input span may overlap `dst`.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "Each Limb product is widened exactly to DoubleLimb, and fixed-width extraction is the measured three-limb hot path"
)]
#[inline(always)]
pub unsafe fn mul_3x3_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // Initialize row zero. For every column, b[j]*a[0] + carry is at most
    // (B-1)^2 + (B-2) = B^2-B-1, so it fits DoubleLimb exactly.
    // SAFETY: the caller provides both three-limb inputs and six output limbs.
    let scalar0 = unsafe { *a } as DoubleLimb;
    let mut carry: DoubleLimb = 0;
    let mut column = 0_usize;
    while column < 3 {
        // SAFETY: column is in 0..3 and the output row fits dst[0..3].
        let source = unsafe { *b.add(column) } as DoubleLimb;
        let product = source.wrapping_mul(scalar0).wrapping_add(carry);
        // SAFETY: column is in 0..3.
        unsafe {
            *dst.add(column) = product as Limb;
        }
        carry = product.wrapping_shr(Limb::BITS);
        column = column.wrapping_add(1);
    }
    // SAFETY: dst has six limbs and index 3 is the closing row-zero carry.
    unsafe {
        *dst.add(3) = carry as Limb;
    }

    // Accumulate rows one and two. A column sum is bounded by
    // (B-1)^2 + (B-1) + (B-2) = B^2-2, again fitting DoubleLimb exactly.
    let mut row = 1_usize;
    while row < 3 {
        // SAFETY: row is 1 or 2, hence a[row] exists.
        let scalar = unsafe { *a.add(row) } as DoubleLimb;
        carry = 0;
        column = 0;
        while column < 3 {
            let output_index = row.wrapping_add(column);
            // SAFETY: column is in 0..3 and output_index is in row..row+3.
            let source = unsafe { *b.add(column) } as DoubleLimb;
            // SAFETY: output_index is 1..=4 and was initialized by the prior
            // row either as a product limb or as its closing carry.
            let existing = unsafe { *dst.add(output_index) } as DoubleLimb;
            let product = source
                .wrapping_mul(scalar)
                .wrapping_add(existing)
                .wrapping_add(carry);
            // SAFETY: output_index is in 1..=4.
            unsafe {
                *dst.add(output_index) = product as Limb;
            }
            carry = product.wrapping_shr(Limb::BITS);
            column = column.wrapping_add(1);
        }
        // SAFETY: row+3 is 4 or 5 and closes the current row.
        unsafe {
            *dst.add(row.wrapping_add(3)) = carry as Limb;
        }
        row = row.wrapping_add(1);
    }
}
