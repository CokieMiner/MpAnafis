//! Portable fixed-width kernels for complete basecase multiplication.

use super::{DoubleLimb, Limb};

#[cfg(not(all(
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_feature = "adx",
    target_feature = "bmi2"
)))]
macro_rules! initialize_fixed_row {
    ($dst:ident, $src:ident, $scalar:ident, $carry_at:literal; $(($src_at:literal, $dst_at:literal)),+ $(,)?) => {{
        let mut carry = DoubleLimb::MIN;
        $(
            // SAFETY: every literal source and destination offset belongs to
            // the fixed spans proved by the enclosing kernel's contract.
            let source = unsafe { *$src.add($src_at) } as DoubleLimb;
            // SAFETY: source and scalar are widened limbs; carry is at most
            // B-1, so their product and sum are below B^2.
            let product = unsafe { source.unchecked_mul($scalar).unchecked_add(carry) };
            // SAFETY: the literal output offset lies in the complete product.
            unsafe {
                *$dst.add($dst_at) = product as Limb;
            }
            carry = product >> Limb::BITS;
        )+
        // SAFETY: the literal closing-carry offset lies in the destination.
        unsafe {
            *$dst.add($carry_at) = carry as Limb;
        }
    }};
}

#[cfg(not(all(
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_feature = "adx",
    target_feature = "bmi2"
)))]
macro_rules! accumulate_fixed_row {
    ($dst:ident, $src:ident, $scalar:ident, $carry_at:literal; $(($src_at:literal, $dst_at:literal)),+ $(,)?) => {{
        let mut carry = DoubleLimb::MIN;
        $(
            // SAFETY: every literal source and destination offset belongs to
            // the fixed spans proved by the enclosing kernel's contract. The
            // destination limb was initialized by the preceding row.
            let source = unsafe { *$src.add($src_at) } as DoubleLimb;
            // SAFETY: the literal destination offset belongs to the initialized
            // overlap left by the preceding fixed row.
            let existing = unsafe { *$dst.add($dst_at) } as DoubleLimb;
            // SAFETY: source, scalar, existing, and carry are widened limbs.
            // Their product-plus-two-addends is <= (B-1)^2+2(B-1)=B^2-1.
            let product = unsafe { source
                .unchecked_mul($scalar)
                .unchecked_add(existing)
                .unchecked_add(carry) };
            // SAFETY: the literal output offset lies in the complete product.
            unsafe {
                *$dst.add($dst_at) = product as Limb;
            }
            carry = product >> Limb::BITS;
        )+
        // SAFETY: the literal closing-carry offset lies in the destination.
        unsafe {
            *$dst.add($carry_at) = carry as Limb;
        }
    }};
}

/// Write the exact product of two two-limb operands.
///
/// # Safety
///
/// `a` and `b` must each be readable for two limbs, `dst` must be writable
/// for four limbs, and neither input span may overlap `dst`.
/// Pointers must be aligned, input limbs initialized; output may be uninitialized.
#[expect(
    clippy::as_conversions,
    clippy::inline_always,
    reason = "The four products are widened to DoubleLimb and fixed-column extraction is the measured two-limb hot path"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "The four products are widened to DoubleLimb and fixed-column extraction is the measured two-limb hot path"
    )
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

    // SAFETY: widened limb products are <= (B-1)^2. Columns one and two
    // are <= 3B-4 and 3B-3, below B^2 on every limb width. Their carries
    // are at most two; the complete product is < B^4, so top fits Limb.
    let (product00, column1, column2, top) = unsafe {
        let product00 = a0.unchecked_mul(b0);
        let product01 = a0.unchecked_mul(b1);
        let product10 = a1.unchecked_mul(b0);
        let product11 = a1.unchecked_mul(b1);
        let column1 = (product00 >> Limb::BITS)
            .unchecked_add(product01 as Limb as DoubleLimb)
            .unchecked_add(product10 as Limb as DoubleLimb);
        let column2 = (product01 >> Limb::BITS)
            .unchecked_add(product10 >> Limb::BITS)
            .unchecked_add(product11 as Limb as DoubleLimb)
            .unchecked_add(column1 >> Limb::BITS);
        let top = (product11 >> Limb::BITS).unchecked_add(column2 >> Limb::BITS);
        (product00, column1, column2, top)
    };

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
/// Pointers must be aligned, input limbs initialized; output may be uninitialized.
#[expect(
    clippy::as_conversions,
    clippy::inline_always,
    reason = "Each Limb product is widened exactly to DoubleLimb, and fixed-width extraction is the measured three-limb hot path"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "Each Limb product is widened exactly to DoubleLimb, and fixed-width extraction is the measured three-limb hot path"
    )
)]
#[inline(always)]
pub unsafe fn mul_3x3_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // Initialize row zero. For every column, b[j]*a[0] + carry is at most
    // (B-1)^2 + (B-2) = B^2-B-1, so it fits DoubleLimb exactly.
    // SAFETY: the caller provides both three-limb inputs and six output limbs.
    let scalar0 = unsafe { *a } as DoubleLimb;
    let mut carry: DoubleLimb = 0;
    for column in 0..3 {
        // SAFETY: column is in 0..3 and the output row fits dst[0..3].
        let source = unsafe { *b.add(column) } as DoubleLimb;
        // SAFETY: widened limbs and the row carry give product+carry < B^2.
        let product = unsafe { source.unchecked_mul(scalar0).unchecked_add(carry) };
        // SAFETY: column is in 0..3.
        unsafe {
            *dst.add(column) = product as Limb;
        }
        carry = product >> Limb::BITS;
    }
    // SAFETY: dst has six limbs and index 3 is the closing row-zero carry.
    unsafe {
        *dst.add(3) = carry as Limb;
    }

    // Accumulate rows one and two. A column sum is bounded by
    // (B-1)^2 + 2(B-1) = B^2-1, again fitting DoubleLimb exactly.
    for row in 1..3 {
        // SAFETY: row is 1 or 2, hence a[row] exists.
        let scalar = unsafe { *a.add(row) } as DoubleLimb;
        carry = 0;
        for column in 0..3 {
            // SAFETY: row <= 2 and column <= 2 give output_index <= 4.
            let output_index = unsafe { row.unchecked_add(column) };
            // SAFETY: column is in 0..3 and output_index is in row..row+3.
            let source = unsafe { *b.add(column) } as DoubleLimb;
            // SAFETY: output_index is 1..=4 and was initialized by the prior
            // row either as a product limb or as its closing carry.
            let existing = unsafe { *dst.add(output_index) } as DoubleLimb;
            // SAFETY: each operand is a widened limb; the column bound above
            // proves every multiplication and addition fits DoubleLimb.
            let product = unsafe { source
                .unchecked_mul(scalar)
                .unchecked_add(existing)
                .unchecked_add(carry) };
            // SAFETY: output_index is in 1..=4.
            unsafe {
                *dst.add(output_index) = product as Limb;
            }
            carry = product >> Limb::BITS;
        }
        // SAFETY: row+3 is 4 or 5 and closes the current row.
        unsafe {
            *dst.add(row.unchecked_add(3)) = carry as Limb;
        }
    }
}

/// Write the exact product of two four-limb operands with every row unrolled.
///
/// # Safety
///
/// `a` and `b` must each be readable for four limbs, `dst` must be writable
/// for eight limbs, and neither input span may overlap `dst`.
/// Pointers must be aligned, input limbs initialized; output may be uninitialized.
#[expect(
    clippy::as_conversions,
    clippy::inline_always,
    reason = "Every product is widened to DoubleLimb and the literal offsets encode exact four-limb spans"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "Every product is widened to DoubleLimb and the literal offsets encode exact four-limb spans"
    )
)]
#[inline(always)]
#[cfg(not(all(
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_feature = "adx",
    target_feature = "bmi2"
)))]
pub unsafe fn mul_4x4_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // A row cell is bounded by (B-1)^2 + 2(B-1) = B^2-1,
    // so every widened multiply-add is exact on every supported limb width.
    // SAFETY: the caller provides the exact four-limb input span.
    let scalar0 = unsafe { *a } as DoubleLimb;
    initialize_fixed_row!(dst, b, scalar0, 4; (0, 0), (1, 1), (2, 2), (3, 3));

    // SAFETY: indices one through three lie in the caller-proven input span.
    let scalar1 = unsafe { *a.add(1) } as DoubleLimb;
    accumulate_fixed_row!(dst, b, scalar1, 5; (0, 1), (1, 2), (2, 3), (3, 4));
    // SAFETY: index two lies in the caller-proven input span.
    let scalar2 = unsafe { *a.add(2) } as DoubleLimb;
    accumulate_fixed_row!(dst, b, scalar2, 6; (0, 2), (1, 3), (2, 4), (3, 5));
    // SAFETY: index three lies in the caller-proven input span.
    let scalar3 = unsafe { *a.add(3) } as DoubleLimb;
    accumulate_fixed_row!(dst, b, scalar3, 7; (0, 3), (1, 4), (2, 5), (3, 6));
}

/// Write the exact product of two eight-limb operands with every row unrolled.
///
/// # Safety
///
/// `a` and `b` must each be readable for eight limbs, `dst` must be writable
/// for sixteen limbs, and neither input span may overlap `dst`.
/// Pointers must be aligned, input limbs initialized; output may be uninitialized.
#[expect(
    clippy::as_conversions,
    clippy::inline_always,
    reason = "Every product is widened to DoubleLimb and the literal offsets encode exact eight-limb spans"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "Every product is widened to DoubleLimb and the literal offsets encode exact eight-limb spans"
    )
)]
#[inline(always)]
#[cfg(not(all(
    not(miri),
    target_arch = "x86_64",
    target_pointer_width = "64",
    target_feature = "adx",
    target_feature = "bmi2"
)))]
pub unsafe fn mul_8x8_portable_unchecked(dst: *mut Limb, a: *const Limb, b: *const Limb) {
    // The same B^2-1 row-cell bound used by the four-limb kernel is independent
    // of row width; the literal geometry removes all loop control and indexing.
    // SAFETY: the caller provides the exact eight-limb input span.
    let scalar0 = unsafe { *a } as DoubleLimb;
    initialize_fixed_row!(
        dst, b, scalar0, 8;
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    );

    // SAFETY: indices one through seven lie in the caller-proven input span.
    let scalar1 = unsafe { *a.add(1) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar1, 9;
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    );
    // SAFETY: index two lies in the caller-proven input span.
    let scalar2 = unsafe { *a.add(2) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar2, 10;
        (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9),
    );
    // SAFETY: index three lies in the caller-proven input span.
    let scalar3 = unsafe { *a.add(3) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar3, 11;
        (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
    );
    // SAFETY: index four lies in the caller-proven input span.
    let scalar4 = unsafe { *a.add(4) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar4, 12;
        (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    );
    // SAFETY: index five lies in the caller-proven input span.
    let scalar5 = unsafe { *a.add(5) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar5, 13;
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12),
    );
    // SAFETY: index six lies in the caller-proven input span.
    let scalar6 = unsafe { *a.add(6) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar6, 14;
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (7, 13),
    );
    // SAFETY: index seven lies in the caller-proven input span.
    let scalar7 = unsafe { *a.add(7) } as DoubleLimb;
    accumulate_fixed_row!(
        dst, b, scalar7, 15;
        (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14),
    );
}
