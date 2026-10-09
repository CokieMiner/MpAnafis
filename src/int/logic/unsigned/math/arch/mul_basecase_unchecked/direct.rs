//! Direct composition of the selected schoolbook multiplication kernels.

use super::{
    ArchKernels, Limb, mul_2x2_portable_unchecked, mul_3x3_portable_unchecked,
    mul_4x4_unchecked, mul_8x8_unchecked,
};

/// Evaluates `dst[0..len_a + len_b] <- a[0..len_a] * b[0..len_b]`.
///
/// Computes the complete schoolbook product of two multi-precision operands.
///
/// # Safety
///
/// - Aligned initialized inputs cover `len_a >= 2` and `len_b > 0` limbs.
/// - Aligned `dst` covers `len_a + len_b` writable limbs, which may be uninitialized.
/// - The length sum fits `usize`, and all byte spans fit `isize::MAX`.
/// - Neither input may overlap the destination; the two inputs may overlap.
#[expect(
    clippy::inline_always,
    reason = "Inlining makes compile-time architecture kernels direct calls inside the quadratic loop"
)]
#[inline(always)]
pub unsafe fn mul_basecase_unchecked(
    dst: *mut Limb,
    a: *const Limb,
    len_a: usize,
    b: *const Limb,
    len_b: usize,
) {
    if len_a == 2 && len_b == 2 {
        // SAFETY: this branch proves both exact input widths and the
        // inherited contract provides the disjoint product span.
        unsafe {
            mul_2x2_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 3 && len_b == 3 {
        // SAFETY: this branch proves both exact input widths and the
        // inherited contract provides the disjoint product span.
        unsafe {
            mul_3x3_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 4 && len_b == 4 {
        // SAFETY: this branch proves both exact input widths and the
        // inherited contract provides the disjoint product span.
        unsafe {
            mul_4x4_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 8 && len_b == 8 {
        // SAFETY: this branch proves both exact input widths and the
        // inherited contract provides the disjoint product span.
        unsafe {
            mul_8x8_unchecked(dst, a, b);
        }
        return;
    }

    let multiply_two = ArchKernels::selected_mul_2_limbs_unchecked();
    // SAFETY: len_a >= 2 and the caller guarantees both complete input and
    // output spans. The write-only kernel initializes the first two rows.
    unsafe {
        multiply_two(dst, b, len_b, *a, *a.add(1));
    }
    let mut index = 2_usize;

    if ArchKernels::prefer_add_mul_2_limbs() {
        let add_mul_two = ArchKernels::selected_add_mul_2_limbs_unchecked();
        // SAFETY: index starts at two and remains <= len_a. The loop admits
        // two rows only when len_a-index >= 2. The complete-product span bounds
        // every sum and pointer offset. The extra overlap limb is set to zero,
        // so existing < B^len_b; adding two rows is < B^(len_b+2), proving the
        // closing high-limb addition exact. The selector proves CPU features.
        unsafe {
        while len_a.unchecked_sub(index) >= 2 {
            let carry_index0 = index.unchecked_add(len_b);
            let carry_index1 = carry_index0.unchecked_add(1);
            // The second overlapping row consumes the first carry position;
            // initialize that one not-yet-written limb before accumulation.
                *dst.add(carry_index0) = 0;
                let (carry0, carry1) = add_mul_two(
                    dst.add(index),
                    b,
                    len_b,
                    *a.add(index),
                    *a.add(index.unchecked_add(1)),
                );
                let existing = *dst.add(carry_index0);
                let (sum, overflow) = existing.overflowing_add(carry0);
                *dst.add(carry_index0) = sum;
                let top = carry1.unchecked_add(Limb::from(overflow));
                *dst.add(carry_index1) = top;
            index = index.unchecked_add(2);
        }
        }
    }

    let add_mul_one = ArchKernels::selected_add_mul_limbs_unchecked();
    for row_index in index..len_a {
        // SAFETY: the remaining row and its carry limb fit the complete
        // product span established by the caller.
        let carry = unsafe { add_mul_one(dst.add(row_index), b, len_b, *a.add(row_index)) };
        // SAFETY: index + len_b < len_a + len_b.
        unsafe {
            *dst.add(row_index.unchecked_add(len_b)) = carry;
        }
    }
}
