//! The Toom-Cook 3 driver: split, evaluate, recurse, interpolate.

use core::cmp::max;

use super::{
    AddMulKernel, ArchKernels, Karatsuba, Limb, MiddleValues, Multiplication, Recursive,
    SQR_TOOM_COOK_THRESHOLD, TOOM_COOK_THRESHOLD,
};

/// Namespace for the three-way Toom-Cook multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom3;

impl Toom3 {
    /// Computes the product of two limb slices using Toom-Cook 3-way multiplication.
    ///
    /// `dst` must hold the full product and `scratch` must have at least
    /// [`Multiplication::toom3_mul_scratch_len`] limbs.
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < TOOM_COOK_THRESHOLD
            || b.len() < TOOM_COOK_THRESHOLD
            || a.len() < 3
            || b.len() < 3
        {
            Karatsuba::mul(dst, a, b, scratch);
            return;
        }
        Self::mul_forced(dst, a, b, scratch);
    }

    /// Executes one Toom-Cook 3 level regardless of the configured crossover.
    ///
    /// Recursive subproducts still use normal tier dispatch, so this entry point
    /// isolates the cost of selecting Toom-3 at the current operand width without
    /// changing production thresholds. Operands shorter than three limbs retain
    /// the Karatsuba/basecase fallback because a three-way split is impossible.
    pub fn mul_forced(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 3 || b.len() < 3 {
            Karatsuba::mul(dst, a, b, scratch);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::toom3_mul_forced_scratch_len(a.len(), b.len()),
            "Toom-3 multiplication scratch is undersized: have {}, need {} for {}x{} limbs",
            scratch.len(),
            Multiplication::toom3_mul_forced_scratch_len(a.len(), b.len()),
            a.len(),
            b.len()
        );
        let split_len = max(a.len(), b.len()).div_ceil(3);
        let (a0, a1, a2) = Self::split_three(a, split_len);
        let (b0, b1, b2) = Self::split_three(b, split_len);
        let product_len = split_len.wrapping_mul(2).wrapping_add(1);
        let eval_len = split_len.wrapping_add(1);

        let ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            eval_b,
            inner,
        } = Self::split_scratch(scratch, product_len, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        let (positive_a, _) = one.split_at_mut(eval_len);
        let (positive_b, _) = two.split_at_mut(eval_len);
        let negative_a = Self::evaluate_one_and_negative_one(positive_a, eval_a, a0, a1, a2);
        let negative_b = Self::evaluate_one_and_negative_one(positive_b, eval_b, b0, b1, b2);
        Self::recursive_mul_evaluation(neg_one, eval_a, eval_b, inner, add_mul_kernel);

        eval_a.copy_from_slice(positive_a);
        eval_b.copy_from_slice(positive_b);
        Self::recursive_mul_evaluation(one, eval_a, eval_b, inner, add_mul_kernel);

        Self::evaluate_two_from_one(eval_a, a1, a2, add_mul_kernel);
        Self::evaluate_two_from_one(eval_b, b1, b2, add_mul_kernel);
        Self::recursive_mul_evaluation(two, eval_a, eval_b, inner, add_mul_kernel);

        let low_product_len = a0.len().wrapping_add(b0.len());
        let high_product_len = if a2.is_empty() || b2.is_empty() {
            0
        } else {
            a2.len().wrapping_add(b2.len())
        };
        let high_offset = split_len.wrapping_mul(4);
        {
            let (zero, _) = dst.split_at_mut(low_product_len);
            Self::mul(zero, a0, b0, inner);
        }
        if high_product_len != 0 {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (infinity, _) = high_and_after.split_at_mut(high_product_len);
            Self::mul(infinity, a2, b2, inner);
        }
        Self::clear_destination_outside_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );

        Self::interpolate_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
            MiddleValues {
                one: &mut *one,
                neg_one: &mut *neg_one,
                two: &mut *two,
                neg_one_negative: negative_a ^ negative_b,
            },
        );
        Self::reconstruct_middle(dst, split_len, neg_one, one, two);
    }

    /// Toom-Cook 3-way squaring specialization with symmetric evaluation.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < SQR_TOOM_COOK_THRESHOLD || a.len() < 3 {
            Karatsuba::sqr(dst, a, scratch);
            return;
        }
        Self::sqr_forced(dst, a, scratch);
    }

    /// Executes one Toom-Cook 3 square level regardless of the crossover.
    ///
    /// Recursive squares still use normal dispatch. Inputs shorter than three
    /// limbs retain the Karatsuba/basecase fallback because a three-way split is
    /// impossible.
    pub fn sqr_forced(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 3 {
            Karatsuba::sqr(dst, a, scratch);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::toom3_sqr_forced_scratch_len(a.len()),
            "Toom-3 squaring scratch is undersized: have {}, need {} for {} limbs",
            scratch.len(),
            Multiplication::toom3_sqr_forced_scratch_len(a.len()),
            a.len()
        );
        let split_len = a.len().div_ceil(3);
        let (a0, a1, a2) = Self::split_three(a, split_len);
        let product_len = split_len.wrapping_mul(2).wrapping_add(1);
        let eval_len = split_len.wrapping_add(1);

        let ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            inner,
            ..
        } = Self::split_scratch(scratch, product_len, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        let (positive, _) = one.split_at_mut(eval_len);
        let _ = Self::evaluate_one_and_negative_one(positive, eval_a, a0, a1, a2);
        Self::recursive_sqr_evaluation(neg_one, eval_a, inner, add_mul_kernel);

        eval_a.copy_from_slice(positive);
        Self::recursive_sqr_evaluation(one, eval_a, inner, add_mul_kernel);
        Self::evaluate_two_from_one(eval_a, a1, a2, add_mul_kernel);
        Self::recursive_sqr_evaluation(two, eval_a, inner, add_mul_kernel);

        let low_product_len = a0.len().wrapping_mul(2);
        let high_product_len = a2.len().wrapping_mul(2);
        let high_offset = split_len.wrapping_mul(4);
        {
            let (zero, _) = dst.split_at_mut(low_product_len);
            Self::sqr(zero, a0, inner);
        }
        if high_product_len != 0 {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (infinity, _) = high_and_after.split_at_mut(high_product_len);
            Self::sqr(infinity, a2, inner);
        }
        Self::clear_destination_outside_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );

        Self::interpolate_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
            MiddleValues {
                one: &mut *one,
                neg_one: &mut *neg_one,
                two: &mut *two,
                neg_one_negative: false,
            },
        );
        Self::reconstruct_middle(dst, split_len, neg_one, one, two);
    }

    /// Toom-3 evaluations carry a guard below seven; recursion stays in this tier.
    fn recursive_mul_evaluation(
        dst: &mut [Limb],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_product::<7, 1>(
            dst,
            evaluation_a,
            evaluation_b,
            scratch,
            add_mul_kernel,
            Self::mul,
        );
    }

    fn recursive_sqr_evaluation(
        dst: &mut [Limb],
        evaluation: &[Limb],
        scratch: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_square::<7, 1>(
            dst,
            evaluation,
            scratch,
            add_mul_kernel,
            Self::sqr,
        );
    }

    fn clear_destination_outside_endpoints(
        dst: &mut [Limb],
        low_product_len: usize,
        high_offset: usize,
        high_product_len: usize,
    ) {
        if high_product_len == 0 {
            let (_, unwritten) = dst.split_at_mut(low_product_len);
            unwritten.fill(0);
            return;
        }

        // Both recursive endpoint products overwrite their complete ranges. Only
        // the gap between them and the tail above the exact infinity product must
        // start at zero before the overlapping coefficients are reconstructed.
        let (before_high, high_and_after) = dst.split_at_mut(high_offset);
        let (_, middle_gap) = before_high.split_at_mut(low_product_len);
        middle_gap.fill(0);
        let (_, trailing_gap) = high_and_after.split_at_mut(high_product_len);
        trailing_gap.fill(0);
    }
}

// ── Scratch layout and the three-way split ────────────────────────────────────

pub struct ScratchLayout<'buffer> {
    pub one: &'buffer mut [Limb],
    pub neg_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
}

impl Toom3 {
    pub const fn split_scratch(
        scratch: &mut [Limb],
        product_len: usize,
        eval_len: usize,
    ) -> ScratchLayout<'_> {
        let (one, after_one) = scratch.split_at_mut(product_len);
        let (neg_one, after_neg_one) = after_one.split_at_mut(product_len);
        let (two, after_two) = after_neg_one.split_at_mut(product_len);
        let (eval_a, after_eval_a) = after_two.split_at_mut(eval_len);
        let (eval_b, inner) = after_eval_a.split_at_mut(eval_len);
        ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            eval_b,
            inner,
        }
    }

    pub const fn local_scratch_len(split_len: usize, inner_space: usize) -> usize {
        let product_len = split_len.wrapping_mul(2).wrapping_add(1);
        let eval_len = split_len.wrapping_add(1);
        product_len
            .wrapping_mul(3)
            .wrapping_add(eval_len.wrapping_mul(2))
            .wrapping_add(inner_space)
    }

    pub const fn split_three(values: &[Limb], split_len: usize) -> (&[Limb], &[Limb], &[Limb]) {
        if values.len() > split_len.wrapping_mul(2) {
            let (part0, after_part0) = values.split_at(split_len);
            let (part1, part2) = after_part0.split_at(split_len);
            (part0, part1, part2)
        } else if values.len() > split_len {
            let (part0, part1) = values.split_at(split_len);
            (part0, part1, &[])
        } else {
            (values, &[], &[])
        }
    }
}
