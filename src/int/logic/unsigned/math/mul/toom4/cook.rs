//! The Toom-Cook 4 driver: split, paired evaluation, interpolate, reconstruct.
use core::cmp::max;

use super::{
    AddMulKernel, ArchKernels, EvaluationBuffers, EvaluationKernels, Limb, MiddleProducts,
    MiddleValues, Multiplication, OperandParts, PointDimensions, Recursive, TierCeiling,
};

/// Namespace for the four-way Toom-Cook multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom4;

pub struct DestinationInterpolation<'buffer> {
    split_len: usize,
    product_len: usize,
    low_product_len: usize,
    high_product_len: usize,
    neg_two: &'buffer mut [Limb],
    neg_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    half: &'buffer mut [Limb],
    neg_two_negative: bool,
    neg_one_negative: bool,
}

impl Toom4 {
    /// Multiply two balanced limb slices with a 4-way Toom-Cook split.
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 4 || b.len() < 4 {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom3);
            return;
        }

        let split_len = max(a.len(), b.len()).div_ceil(4);
        if !Multiplication::operand_has_four_parts(a.len(), split_len)
            || !Multiplication::operand_has_four_parts(b.len(), split_len)
        {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom3);
            return;
        }
        let eval_len = split_len.wrapping_add(1);
        let product_len = eval_len.wrapping_mul(2);
        let ScratchLayout {
            neg_two: at_neg_two,
            neg_one: at_neg_one,
            two: at_two,
            half: at_half,
            eval_a,
            eval_b,
            inner: inner_scratch,
        } = Self::split_scratch(scratch, product_len, eval_len);

        let (a0, a1, a2, a3) = Self::split_four(a, split_len);
        let (b0, b1, b2, b3) = Self::split_four(b, split_len);
        let mut products = MiddleProducts {
            neg_two: at_neg_two,
            neg_one: at_neg_one,
            two: at_two,
            half: at_half,
        };
        let mut evaluations = EvaluationBuffers {
            a: eval_a,
            b: eval_b,
            inner: inner_scratch,
            kernels: EvaluationKernels {
                fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
                add_mul: ArchKernels::selected_add_mul_limbs_unchecked(),
            },
        };
        let (neg_two_negative, neg_one_negative) = Self::multiply_points(
            dst,
            PointDimensions {
                split: split_len,
                evaluation: eval_len,
                product: product_len,
            },
            OperandParts {
                zero: a0,
                one: a1,
                two: a2,
                three: a3,
            },
            OperandParts {
                zero: b0,
                one: b1,
                two: b2,
                three: b3,
            },
            &mut products,
            &mut evaluations,
        );

        let low_product_len = a0.len().wrapping_add(b0.len());
        let high_product_len = if a3.is_empty() || b3.is_empty() {
            0
        } else {
            a3.len().wrapping_add(b3.len())
        };
        let high_offset = split_len.wrapping_mul(6);
        {
            let (at_zero, _) = dst.split_at_mut(low_product_len);
            Recursive::recursive_mul(at_zero, a0, b0, evaluations.inner, TierCeiling::Toom3);
        }
        if high_product_len != 0 {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (at_infinity, _) = high_and_after.split_at_mut(high_product_len);
            Recursive::recursive_mul(at_infinity, a3, b3, evaluations.inner, TierCeiling::Toom3);
        }
        clear_destination_outside_products(
            dst,
            split_len,
            product_len,
            low_product_len,
            high_product_len,
        );

        interpolate_destination(
            dst,
            DestinationInterpolation {
                split_len,
                product_len,
                low_product_len,
                high_product_len,
                neg_two: products.neg_two,
                neg_one: products.neg_one,
                two: products.two,
                half: products.half,
                neg_two_negative,
                neg_one_negative,
            },
        );
    }

    /// Square a limb slice with a 4-way Toom-Cook split.
    #[allow(
        clippy::too_many_lines,
        reason = "balanced paired-evaluation sqr layout"
    )]
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 4 {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom3);
            return;
        }

        let split_len = a.len().div_ceil(4);
        if !Multiplication::operand_has_four_parts(a.len(), split_len) {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom3);
            return;
        }
        let eval_len = split_len.wrapping_add(1);
        let product_len = eval_len.wrapping_mul(2);
        debug_assert!(
            dst.len() >= a.len().wrapping_mul(2),
            "Toom-4 squaring output is shorter than the full square"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom4_sqr_scratch_len(a.len()),
            "Toom-4 squaring scratch buffer is undersized"
        );

        let ScratchLayout {
            neg_two: at_neg_two,
            neg_one: at_neg_one,
            two: at_two,
            half: at_half,
            eval_a: eval,
            inner: inner_scratch,
            ..
        } = Self::split_scratch(scratch, product_len, eval_len);

        let (a0, a1, a2, a3) = Self::split_four(a, split_len);
        let one_offset = split_len.wrapping_mul(2);

        let kernels = EvaluationKernels {
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
            add_mul: ArchKernels::selected_add_mul_limbs_unchecked(),
        };

        let (positive_eval, _) = at_half.split_at_mut(eval_len);
        if Self::evaluate_positive_and_negative_balanced::<true>(
            positive_eval,
            eval,
            a0,
            a1,
            a2,
            a3,
            kernels,
        )
        .is_none()
        {
            Self::evaluate_positive(positive_eval, a0, a1, a2, a3, 1);
            let (temp, _) = at_neg_two.split_at_mut(eval_len);
            let _ = Self::evaluate_negative_magnitude(eval, temp, a0, a1, a2, a3, 1);
        }
        Self::recursive_sqr_evaluation(at_neg_two, eval, inner_scratch, kernels.add_mul);
        eval.copy_from_slice(positive_eval);
        Self::recursive_sqr_evaluation(at_two, eval, inner_scratch, kernels.add_mul);

        Self::evaluate_half_scaled(eval, a0, a1, a2, a3);
        Self::recursive_sqr_evaluation(at_half, eval, inner_scratch, kernels.add_mul);

        {
            let (_, one_and_after) = dst.split_at_mut(one_offset);
            let (at_one, _) = one_and_after.split_at_mut(product_len);
            let (at_one_eval, _) = at_one.split_at_mut(eval_len);
            if Self::evaluate_positive_and_negative_balanced::<false>(
                at_one_eval,
                eval,
                a0,
                a1,
                a2,
                a3,
                kernels,
            )
            .is_none()
            {
                Self::evaluate_positive(at_one_eval, a0, a1, a2, a3, 0);
                let (temp, _) = at_neg_one.split_at_mut(eval_len);
                let _ = Self::evaluate_negative_magnitude(eval, temp, a0, a1, a2, a3, 0);
            }
            Self::recursive_sqr_evaluation(at_neg_one, eval, inner_scratch, kernels.add_mul);
            eval.copy_from_slice(at_one_eval);
            Self::recursive_sqr_evaluation(at_one, eval, inner_scratch, kernels.add_mul);
        }

        let low_product_len = a0.len().wrapping_mul(2);
        let high_product_len = a3.len().wrapping_mul(2);
        let high_offset = split_len.wrapping_mul(6);
        {
            let (at_zero, _) = dst.split_at_mut(low_product_len);
            Recursive::recursive_sqr(at_zero, a0, inner_scratch, TierCeiling::Toom3);
        }
        if high_product_len != 0 {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (at_infinity, _) = high_and_after.split_at_mut(high_product_len);
            Recursive::recursive_sqr(at_infinity, a3, inner_scratch, TierCeiling::Toom3);
        }
        clear_destination_outside_products(
            dst,
            split_len,
            product_len,
            low_product_len,
            high_product_len,
        );

        interpolate_destination(
            dst,
            DestinationInterpolation {
                split_len,
                product_len,
                low_product_len,
                high_product_len,
                neg_two: at_neg_two,
                neg_one: at_neg_one,
                two: at_two,
                half: at_half,
                neg_two_negative: false,
                neg_one_negative: false,
            },
        );
    }
}

fn interpolate_destination(dst: &mut [Limb], values: DestinationInterpolation<'_>) {
    let DestinationInterpolation {
        split_len,
        product_len,
        low_product_len,
        high_product_len,
        neg_two,
        neg_one,
        two,
        half,
        neg_two_negative,
        neg_one_negative,
    } = values;
    // Evaluation products need only 2m+1 active limbs: each guard is below 15,
    // so the product guard is below 225. Preserve the extra physical limb for
    // favorable scratch alignment, but exclude it from every linear pass.
    let active_product_len = product_len.saturating_sub(1);
    let (active_neg_two, _) = neg_two.split_at_mut(active_product_len);
    let (active_neg_one, _) = neg_one.split_at_mut(active_product_len);
    let (active_two, _) = two.split_at_mut(active_product_len);
    let (active_half, _) = half.split_at_mut(active_product_len);
    let one_offset = split_len.wrapping_mul(2);
    let high_offset = split_len.wrapping_mul(6);
    {
        let (before_one, one_and_after) = dst.split_at_mut(one_offset);
        let (at_one_storage, after_one) = one_and_after.split_at_mut(product_len);
        let (at_one, _) = at_one_storage.split_at_mut(active_product_len);
        let (at_zero, _) = before_one.split_at(low_product_len);
        if high_product_len == 0 {
            Toom4::interpolate_with_endpoints(
                at_zero,
                &[],
                MiddleValues {
                    neg_two: &mut *active_neg_two,
                    one: at_one,
                    neg_one: &mut *active_neg_one,
                    two: &mut *active_two,
                    half: &mut *active_half,
                    neg_two_negative,
                    neg_one_negative,
                },
            );
        } else {
            let after_one_offset = one_offset.wrapping_add(product_len);
            let infinity_offset = high_offset.saturating_sub(after_one_offset);
            let (_, infinity_and_after) = after_one.split_at(infinity_offset);
            let (at_infinity, _) = infinity_and_after.split_at(high_product_len);
            Toom4::interpolate_with_endpoints(
                at_zero,
                at_infinity,
                MiddleValues {
                    neg_two: &mut *active_neg_two,
                    one: at_one,
                    neg_one: &mut *active_neg_one,
                    two: &mut *active_two,
                    half: &mut *active_half,
                    neg_two_negative,
                    neg_one_negative,
                },
            );
        }
    }
    Toom4::reconstruct_around_quadratic(
        dst,
        split_len,
        active_neg_two,
        active_neg_one,
        active_two,
        active_half,
    );
}

impl Toom4 {
    /// Toom-4 evaluation guards are below fifteen, so the guard product is below
    /// 225 and fits one limb on every supported target. The evaluated product may
    /// re-enter Toom-4; its endpoint squares may not.
    pub fn recursive_mul_evaluation(
        dst: &mut [Limb],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_product::<15, 1>(
            dst,
            evaluation_a,
            evaluation_b,
            scratch,
            kernel,
            |product, low_a, low_b, inner| {
                Recursive::recursive_mul(product, low_a, low_b, inner, TierCeiling::Toom4);
            },
        );
    }

    pub fn recursive_sqr_evaluation(
        dst: &mut [Limb],
        evaluation: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_square::<15, 1>(
            dst,
            evaluation,
            scratch,
            kernel,
            |square, low, inner| {
                Recursive::recursive_sqr(square, low, inner, TierCeiling::Toom3);
            },
        );
    }
}

fn clear_destination_outside_products(
    dst: &mut [Limb],
    split_len: usize,
    product_len: usize,
    low_product_len: usize,
    high_product_len: usize,
) {
    let one_offset = split_len.wrapping_mul(2);
    let high_offset = split_len.wrapping_mul(6);
    let (before_one, one_and_after) = dst.split_at_mut(one_offset);
    let (_, gap_before_one) = before_one.split_at_mut(low_product_len);
    gap_before_one.fill(0);
    let (_, after_one) = one_and_after.split_at_mut(product_len);

    if high_product_len == 0 {
        after_one.fill(0);
        return;
    }

    // W(0), W(1), and W(infinity) overwrite their complete destination
    // ranges. Clear only the two disjoint gaps that interpolation will add
    // into; this preserves the same zero-base invariant as a full fill.
    let after_one_offset = one_offset.wrapping_add(product_len);
    let gap_after_one_len = high_offset.wrapping_sub(after_one_offset);
    let (gap_after_one, infinity_and_after) = after_one.split_at_mut(gap_after_one_len);
    gap_after_one.fill(0);
    let (_, trailing_gap) = infinity_and_after.split_at_mut(high_product_len);
    trailing_gap.fill(0);
}

// ── Scratch layout ────────────────────────────────────────────────────────────

pub struct ScratchLayout<'buffer> {
    pub neg_two: &'buffer mut [Limb],
    pub neg_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
}

impl Toom4 {
    pub const fn split_scratch(
        scratch: &mut [Limb],
        product_len: usize,
        eval_len: usize,
    ) -> ScratchLayout<'_> {
        let (neg_two, after_neg_two) = scratch.split_at_mut(product_len);
        let (neg_one, after_neg_one) = after_neg_two.split_at_mut(product_len);
        let (two, after_two) = after_neg_one.split_at_mut(product_len);
        let (half, after_half) = after_two.split_at_mut(product_len);
        let (eval_a, after_eval_a) = after_half.split_at_mut(eval_len);
        let (eval_b, inner) = after_eval_a.split_at_mut(eval_len);

        ScratchLayout {
            neg_two,
            neg_one,
            two,
            half,
            eval_a,
            eval_b,
            inner,
        }
    }

    pub const fn local_scratch_len(split_len: usize, inner_space: usize) -> usize {
        let eval_len = split_len.wrapping_add(1);
        let product_len = eval_len.wrapping_mul(2);
        product_len
            .wrapping_mul(4)
            .wrapping_add(eval_len.wrapping_mul(2))
            .wrapping_add(inner_space)
    }
}
