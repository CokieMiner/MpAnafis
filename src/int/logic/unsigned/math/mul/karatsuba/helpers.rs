//! Fixed-width evaluation and reconstruction helpers for Karatsuba.

use core::ptr::copy_nonoverlapping;

use super::{Addition, ArchKernels, Karatsuba, Limb};

impl Karatsuba {
    /// Write `|left-right|` and return whether the mathematical difference is negative.
    pub fn abs_difference_equal_width(dst: &mut [Limb], left: &[Limb], right: &[Limb]) -> bool {
        debug_assert_eq!(left.len(), right.len(), "difference widths must match");
        let mut index = left.len();
        let mut left_is_less = false;
        while index > 0 {
            index = index.wrapping_sub(1);
            // SAFETY: index was decremented from a positive value no greater than
            // the common input length.
            let left_limb = unsafe { *left.get_unchecked(index) };
            // SAFETY: right has the same length as left.
            let right_limb = unsafe { *right.get_unchecked(index) };
            if left_limb != right_limb {
                left_is_less = left_limb < right_limb;
                break;
            }
        }

        let (minuend, subtrahend) = if left_is_less {
            (right, left)
        } else {
            (left, right)
        };
        // SAFETY: all three slices have the same width and the destination does
        // not overlap either source.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                dst.as_mut_ptr(),
                minuend.as_ptr(),
                subtrahend.as_ptr(),
                minuend.len(),
            )
        };
        debug_assert_eq!(borrow, 0, "absolute difference underflowed");
        left_is_less
    }

    /// Add one endpoint product into a fixed-width middle coefficient.
    pub fn add_product_to_middle(middle: &mut [Limb], product: &[Limb]) {
        let carry = Addition::add_slice_in_place(middle, product);
        let (_, carry_and_guard) = middle.split_at_mut(product.len());
        let Some((carry_limb, remaining)) = carry_and_guard.split_first_mut() else {
            debug_assert_eq!(carry, 0, "endpoint addition lost its carry limb");
            return;
        };
        let (sum, mut overflow) = carry_limb.overflowing_add(carry);
        *carry_limb = sum;
        // Equal-width endpoints end at the fixed guard and cannot overflow it
        // because the final cross coefficient fits. Odd-width high endpoints are
        // shorter; in that case a carry must ripple through the intervening limbs.
        for limb in remaining {
            if !overflow {
                break;
            }
            let (next, next_overflow) = limb.overflowing_add(1);
            *limb = next;
            overflow = next_overflow;
        }
        // A final overflow is the modular cancellation of the all-ones sign
        // extension installed by `reverse_subtract_product_from_middle`. Dropping
        // it is exact because the reconstructed non-negative coefficient fits the
        // retained fixed width.
    }

    /// Replace `middle` with `product - middle`, sign-extended through its guard.
    pub fn reverse_subtract_product_from_middle(middle: &mut [Limb], product: &[Limb]) {
        debug_assert_eq!(
            middle.len(),
            product.len().wrapping_add(1),
            "middle guard is missing"
        );
        let (body, guard) = middle.split_at_mut(product.len());
        // SAFETY: body and product have equal lengths and are disjoint. The
        // subtraction backend loads the aliased subtrahend limb before writing
        // the destination limb, so using body as both dst and src2 is valid.
        let borrow = unsafe {
            ArchKernels::sub_limbs_3_unchecked(
                body.as_mut_ptr(),
                product.as_ptr(),
                body.as_ptr(),
                product.len(),
            )
        };
        // `product-middle` may be negative before the other endpoint is added.
        // Extending the final borrow with all ones preserves that signed value
        // modulo the full guarded width.
        // SAFETY: every caller allocates one guard limb, so guard is non-empty.
        let top = unsafe { guard.first_mut().unwrap_unchecked() };
        *top = Limb::MIN.wrapping_sub(borrow);
    }

    #[allow(clippy::inline_always, reason = "Critical for Karatsuba evaluation")]
    #[inline(always)]
    pub fn add_slices_in_place(dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> usize {
        let (longer, shorter) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let long_len = longer.len();
        let short_len = shorter.len();

        if long_len > 0 {
            // SAFETY: every caller allocates one guard limb beyond the longer input.
            unsafe {
                copy_nonoverlapping(longer.as_ptr(), dst.as_mut_ptr(), long_len);
            }
        }
        if short_len == 0 {
            return long_len;
        }

        let mut carry = Addition::add_slice_in_place(dst, shorter);
        let mut index = short_len;
        while index < long_len && carry != 0 {
            // SAFETY: index < long_len <= dst.len().
            let limb = unsafe { dst.get_unchecked_mut(index) };
            let (sum, overflow) = limb.overflowing_add(carry);
            *limb = sum;
            carry = Limb::from(overflow);
            index = index.wrapping_add(1);
        }

        // A sum carry is exactly zero or one. Storing it unconditionally avoids
        // an unpredictable branch while directly extending the active length.
        // SAFETY: every caller provides the guard limb at long_len.
        unsafe {
            *dst.get_unchecked_mut(long_len) = carry;
        }
        long_len.wrapping_add(carry)
    }

    #[allow(clippy::inline_always, reason = "Critical for Karatsuba interpolation")]
    #[inline(always)]
    pub fn sub_slices_in_place(dst: &mut [Limb], dst_len: &mut usize, subtrahend: &[Limb]) {
        let subtrahend_len = subtrahend.len();
        if subtrahend_len == 0 {
            return;
        }
        while *dst_len < subtrahend_len {
            // SAFETY: subtrahend_len is bounded by the caller's middle buffer.
            unsafe {
                *dst.get_unchecked_mut(*dst_len) = 0;
            }
            *dst_len = dst_len.wrapping_add(1);
        }

        let borrow = Addition::sub_slice_in_place(dst, subtrahend);
        if borrow != 0 {
            for index in subtrahend_len..*dst_len {
                // SAFETY: index is below the tracked active destination length.
                let limb = unsafe { dst.get_unchecked_mut(index) };
                let (difference, underflow) = limb.overflowing_sub(1);
                *limb = difference;
                if !underflow {
                    break;
                }
            }
        }
        while *dst_len > 0 {
            let index = dst_len.wrapping_sub(1);
            // SAFETY: index < *dst_len <= dst.len().
            if unsafe { *dst.get_unchecked(index) } != 0 {
                break;
            }
            *dst_len = index;
        }
    }
}
