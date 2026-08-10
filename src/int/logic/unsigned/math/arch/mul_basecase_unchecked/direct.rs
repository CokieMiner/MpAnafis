//! Direct composition of the selected schoolbook multiplication kernels.

use super::{ArchKernels, Limb, mul_2x2_portable_unchecked, mul_3x3_portable_unchecked};

/// Compute the complete product of two nonempty limb spans.
///
/// # Safety
///
/// - `a` is readable for `len_a >= 2` limbs.
/// - `b` is readable for `len_b > 0` limbs.
/// - `dst` is writable for `len_a + len_b` limbs.
/// - Neither input span overlaps the destination span.
#[allow(
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
    debug_assert!(len_a >= 2, "basecase outer operand needs two limbs");
    debug_assert!(len_b > 0, "basecase inner operand must be nonempty");

    if len_a == 2 && len_b == 2 {
        // SAFETY: this branch proves both two-limb inputs, and the inherited
        // basecase contract provides a disjoint four-limb destination.
        unsafe {
            mul_2x2_portable_unchecked(dst, a, b);
        }
        return;
    }
    if len_a == 3 && len_b == 3 {
        // SAFETY: this branch proves both three-limb inputs, and the inherited
        // basecase contract provides a disjoint six-limb destination.
        unsafe {
            mul_3x3_portable_unchecked(dst, a, b);
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
        while index.wrapping_add(1) < len_a {
            let carry_index0 = index.wrapping_add(len_b);
            let carry_index1 = carry_index0.wrapping_add(1);
            // The second overlapping row consumes the first carry position;
            // initialize that one not-yet-written limb before accumulation.
            // SAFETY: index+1 < len_a and every pointer remains in its proven span.
            unsafe {
                *dst.add(carry_index0) = 0;
                let (carry0, carry1) = add_mul_two(
                    dst.add(index),
                    b,
                    len_b,
                    *a.add(index),
                    *a.add(index.wrapping_add(1)),
                );
                let existing = *dst.add(carry_index0);
                let (sum, overflow) = existing.overflowing_add(carry0);
                *dst.add(carry_index0) = sum;
                let (top, top_overflow) = carry1.overflowing_add(Limb::from(overflow));
                debug_assert!(!top_overflow, "dual-row carry exceeded the result width");
                *dst.add(carry_index1) = top;
            }
            index = index.wrapping_add(2);
        }
    }

    let add_mul_one = ArchKernels::selected_add_mul_limbs_unchecked();
    while index < len_a {
        // SAFETY: the remaining row and its carry limb fit the complete
        // product span established by the caller.
        let carry = unsafe { add_mul_one(dst.add(index), b, len_b, *a.add(index)) };
        // SAFETY: index + len_b < len_a + len_b.
        unsafe {
            *dst.add(index.wrapping_add(len_b)) = carry;
        }
        index = index.wrapping_add(1);
    }
}
