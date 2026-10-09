//! Exact high products from shortened input suffixes.
//!
//! Set c=skip-2. Dropping max(0,c-n+1) low limbs of A and
//! max(0,c-m+1) of B preserves every scalar product with i+j>=c.
//! Each retained factor has at most total-skip+1 limbs. The omitted
//! polynomial E satisfies E<r*(B-1)*B^c, where r=min(c,m,n).
//! Its carry into the second guard is at most r. An ambiguous guard
//! requires the complete product; otherwise every limb above both guards is exact.

#![expect(
    unsafe_code,
    reason = "The high-product dispatcher proves nonempty suffixes, exact guard offsets, and sufficient disjoint output capacity"
)]

use super::{HighProduct, Limb, MulScratch, Multiplication, ScratchBuffer};

impl HighProduct {
    /// Returns the certified suffix of a product after removing irrelevant input limbs.
    ///
    /// Inputs are nonempty and initialized, `total=a.len()+b.len()`, and
    /// `3<=skip<total`. The operands, output, and scratch are disjoint.
    /// The returned suffix has `total-skip` initialized limbs and one reserved carry slot.
    pub fn trimmed_high_product<'output>(
        a: &[Limb],
        b: &[Limb],
        skip: usize,
        total: usize,
        output: &'output mut ScratchBuffer,
        mul_scratch: &mut MulScratch,
    ) -> &'output mut [Limb] {
        debug_assert!(
            skip >= 3 && skip < total,
            "two low guards precede the retained suffix"
        );
        // SAFETY: nonempty inputs bound both decrements. skip>=3 gives cut>0;
        // skip<total leaves nonempty suffixes and bounds their dropped sum by cut.
        // The retained product contains total-skip+2 limbs after offset. The
        // input byte bounds leave room for its complete width and one carry slot.
        let (cut, left, right, offset, product_len, capacity) = unsafe {
            let cut = skip.unchecked_sub(2);
            let left_drop = cut.saturating_sub(b.len().unchecked_sub(1));
            let right_drop = cut.saturating_sub(a.len().unchecked_sub(1));
            let left = a.get_unchecked(left_drop..);
            let right = b.get_unchecked(right_drop..);
            let product_len = left.len().unchecked_add(right.len());
            (
                cut,
                left,
                right,
                cut.unchecked_sub(left_drop).unchecked_sub(right_drop),
                product_len,
                product_len.unchecked_add(1),
            )
        };
        output.reset_with_capacity(capacity);
        // SAFETY: the original disjoint inputs supply nonempty initialized
        // suffixes; the reservation covers their complete product and carry slot.
        let product = unsafe {
            Multiplication::mul_nonempty_distinct_into_uninit(
                left,
                right,
                output.spare_capacity_mut().get_unchecked_mut(..product_len),
                mul_scratch,
            )
        };
        let bound = cut.min(a.len()).min(b.len());
        // SAFETY: materialized input widths bound r below B. The initialized
        // product contains both guards after offset; guard+r<B certifies the suffix.
        let ambiguous = unsafe {
            *product.get_unchecked(offset.unchecked_add(1)) > Limb::MAX.unchecked_sub(bound)
        };
        if ambiguous {
            // SAFETY: the input byte bounds leave room for total+1 native limbs.
            let full_capacity = unsafe { total.unchecked_add(1) };
            output.reset_with_capacity(full_capacity);
            // SAFETY: the reservation covers the original disjoint full product.
            // Every limb is initialized before set_len exposes the exact suffix.
            unsafe {
                let _ = Multiplication::mul_nonempty_distinct_into_uninit(
                    a,
                    b,
                    output.spare_capacity_mut().get_unchecked_mut(..total),
                    mul_scratch,
                );
                output.set_len(total);
                return output.get_unchecked_mut(skip..);
            }
        }
        // SAFETY: the shortened product initialized product_len limbs.
        // Certification proves its suffix after both guards is exact.
        // The reserved carry slot remains outside the initialized length.
        unsafe {
            output.set_len(product_len);
            output.get_unchecked_mut(offset.unchecked_add(2)..)
        }
    }
}
