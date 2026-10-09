//! Fixed-width Karatsuba products, evaluation and middle-coefficient kernels.

#![expect(
    unsafe_code,
    reason = "Equal-width difference buffers and guarded reconstruction layouts bound initialized raw spans and carry propagation"
)]

use super::{Addition, ArchKernels, Karatsuba, Limb, LimbOutput, Schoolbook, SharedEval};

impl Karatsuba {
    /// Exact 32-limb Karatsuba using fixed-width absolute differences.
    ///
    /// With `z0 = a0*b0`, `z2 = a1*b1` and `d = (a0-a1)(b0-b1)`,
    /// the middle coefficient is `z0+z2-d`. Absolute differences retain
    /// three 16-limb leaves. The signed difference cancels modulo `B^33`;
    /// the nonnegative cross coefficient is strictly below `B^33`.
    pub fn mul_balanced_32(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        const SPLIT: usize = 16;
        const PRODUCT_LIMBS: usize = 32;
        const MIDDLE_LIMBS: usize = 33;

        // SAFETY: the parent selects two 32-limb inputs; their two 16-limb
        // halves produce disjoint 32-limb endpoints in a 64-limb destination.
        let ((a0, a1), (b0, b1), (low_product, high_product)) = unsafe {
            let (low, after_low) = dst.split_at_mut_unchecked(PRODUCT_LIMBS);
            let (high, _) = after_low.split_at_mut_unchecked(PRODUCT_LIMBS);
            (
                a.split_at_unchecked(SPLIT),
                b.split_at_unchecked(SPLIT),
                (low, high),
            )
        };
        Schoolbook::mul_fixed_equal_distinct::<16>(low_product, a0, b0);
        Schoolbook::mul_fixed_equal_distinct::<16>(high_product, a1, b1);
        // SAFETY: both fixed kernels initialized their complete 32-limb endpoints.
        let (low_value, high_value) = unsafe {
            (
                LimbOutput::assume_init_mut(low_product),
                LimbOutput::assume_init_mut(high_product),
            )
        };
        // SAFETY: the validated 65-limb layout reserves 16+16 difference
        // limbs and 33 middle limbs in consecutive disjoint partitions.
        let (a_difference, b_difference, middle_product) = unsafe {
            let (a_diff, after_a) = scratch.split_at_mut_unchecked(SPLIT);
            let (b_diff, after_b) = after_a.split_at_mut_unchecked(SPLIT);
            let (middle, _) = after_b.split_at_mut_unchecked(MIDDLE_LIMBS);
            (a_diff, b_diff, middle)
        };
        let a_negative = Self::abs_difference_equal_width(a_difference, a0, a1);
        let b_negative = Self::abs_difference_equal_width(b_difference, b0, b1);
        // SAFETY: the 33-limb middle retains one guard above the 32-limb product.
        let (middle_value, middle_guard) =
            unsafe { middle_product.split_at_mut_unchecked(PRODUCT_LIMBS) };
        Schoolbook::mul_fixed_equal_distinct::<16>(middle_value, a_difference, b_difference);
        if a_negative == b_negative {
            // Reverse subtraction initializes the guard from the borrow.
            Self::reverse_subtract_product_from_middle(middle_product, low_value);
        } else {
            // Addition needs a zero guard above the exact 32-limb product.
            // SAFETY: the 32-limb layout reserves 33 limbs for the middle.
            unsafe {
                *middle_guard.get_unchecked_mut(0) = 0;
            }
            Self::add_product_to_middle(middle_product, low_value);
        }
        Self::add_product_to_middle(middle_product, high_value);
        let middle_len = SharedEval::active_len(middle_product);
        // SAFETY: active_len returns a prefix length inside middle_product.
        let (active_middle, _) = unsafe { middle_product.split_at_unchecked(middle_len) };
        debug_assert!(
            SPLIT <= dst.len() && active_middle.len() <= dst.len().saturating_sub(SPLIT),
            "32-limb Karatsuba middle coefficient exceeds the destination"
        );
        // SAFETY: both endpoint writers cover all 64 exact output limbs. The
        // normalized middle is their exact 16-limb-shifted cross coefficient;
        // its bounded addition cannot leave the complete product destination.
        let _ = unsafe {
            let initialized = LimbOutput::assume_init_mut(dst.get_unchecked_mut(..64));
            SharedEval::fused_add_shifted_in_place(initialized, active_middle, SPLIT)
        };
    }

    /// Write `|left-right|` and return whether the mathematical difference is negative.
    pub fn abs_difference_equal_width(dst: &mut [Limb], left: &[Limb], right: &[Limb]) -> bool {
        debug_assert_eq!(left.len(), right.len(), "difference widths must match");
        debug_assert_eq!(dst.len(), left.len(), "difference destination must match");
        let mut index = left.len();
        let mut left_is_less = false;
        while index > 0 {
            // SAFETY: the loop condition proves index > 0.
            index = unsafe { index.unchecked_sub(1) };
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
    ///
    /// Both endpoint factors are nonempty, so the product has at least two
    /// limbs. Every middle coefficient retains a guard above either endpoint.
    pub fn add_product_to_middle(middle: &mut [Limb], product: &[Limb]) {
        debug_assert!(product.len() >= 2, "both endpoint factors are nonempty");
        debug_assert!(middle.len() > product.len(), "the middle retains a guard");
        // SAFETY: each Karatsuba split supplies an initialized endpoint of at
        // least two limbs and a disjoint, longer initialized middle buffer.
        // The architecture facade selects a supported backend for these spans.
        let carry = unsafe {
            ArchKernels::add_limbs_unchecked(middle.as_mut_ptr(), product.as_ptr(), product.len())
        };
        // SAFETY: middle has 2*split+1 limbs and each endpoint has at most
        // 2*split limbs. The suffix beginning at product.len() is nonempty.
        let (carry_limb, remaining) = unsafe {
            middle
                .get_unchecked_mut(product.len()..)
                .split_first_mut()
                .unwrap_unchecked()
        };
        let (sum, mut overflow) = carry_limb.overflowing_add(carry);
        *carry_limb = sum;
        // A full-width endpoint reaches the guard immediately. A shorter high
        // endpoint can first carry through the intervening middle limbs.
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
            // SAFETY: a valid product slice spans at most isize::MAX bytes,
            // with Limb >= 2 bytes; product.len()+1 fits every pointer width.
            unsafe { product.len().unchecked_add(1) },
            "middle guard is missing"
        );
        // SAFETY: every difference-form reconstruction retains exactly one
        // initialized middle guard above the complete endpoint product.
        let (body, guard) = unsafe { middle.split_at_mut_unchecked(product.len()) };
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

    /// Adds a full low block and its nonempty high block into separate storage.
    ///
    /// The Karatsuba split guarantees `a.len() >= b.len() > 0`; the destination
    /// has `a.len() + 1` limbs. Each body limb is written once, followed by the
    /// binary guard carry.
    #[expect(
        clippy::inline_always,
        reason = "Evaluation callers reuse the returned carry length without a separate call frame"
    )]
    #[inline(always)]
    pub fn add_blocks(dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> usize {
        debug_assert!(
            !b.is_empty() && a.len() >= b.len(),
            "the Karatsuba low block contains the nonempty high block width"
        );
        debug_assert!(dst.len() > a.len(), "the sum needs one guard limb");
        let long_len = a.len();
        let short_len = b.len();
        // SAFETY: the validated split supplies two nonempty, initialized inputs
        // with at least short_len limbs and a disjoint writable destination.
        let common_carry = unsafe {
            ArchKernels::add_limbs_3_unchecked(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), short_len)
        };
        // SAFETY: the split proves short_len <= long_len, so the subtraction
        // exists. The disjoint tail spans lie within a and the guarded dst.
        let carry = unsafe {
            let tail_len = long_len.unchecked_sub(short_len);
            Addition::copy_tail_with_carry(
                dst.as_mut_ptr().add(short_len),
                a.as_ptr().add(short_len),
                tail_len,
                common_carry,
            )
        };

        // A sum carry is exactly zero or one. Storing it unconditionally avoids
        // an unpredictable branch while directly extending the active length.
        // SAFETY: every caller provides the guard limb at long_len.
        unsafe {
            *dst.get_unchecked_mut(long_len) = carry;
        }
        // SAFETY: carry is binary and the destination contains long_len+1
        // initialized limbs, bounding the active sum length by dst.len().
        unsafe { long_len.unchecked_add(carry) }
    }
}
