//! The Toom-Cook 4 driver: split, paired evaluation, interpolate, reconstruct.

#![expect(
    unsafe_code,
    reason = "Four-part admission bounds disjoint product slots, initialized endpoints, and guarded interpolation spans"
)]

use core::cmp::max;

use super::{
    AddMulKernel, ArchKernels, EvaluationBuffers, EvaluationKernels, Limb, LimbOutput,
    MiddleProducts, MiddleValues, Multiplication, OperandParts, PointDimensions, Recursive,
    TierCeiling,
};

/// Namespace for the four-way Toom-Cook multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom4;

struct DestinationInterpolation<'buffer> {
    split_len: usize,
    product_len: usize,
    high_product_len: usize,
    neg_two: &'buffer mut [Limb],
    neg_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    half: &'buffer mut [Limb],
    neg_two_negative: bool,
    neg_one_negative: bool,
}

struct ScratchLayout<'buffer> {
    neg_two: &'buffer mut [Limb],
    neg_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    half: &'buffer mut [Limb],
    eval_a: &'buffer mut [Limb],
    eval_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

impl Toom4 {
    /// Multiply two balanced limb slices with a 4-way Toom-Cook split.
    pub fn mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
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
        // SAFETY: admitted operands have |a|,|b|>3m, so their full product
        // contains both 2m+2-limb W(1) storage at 2m and the endpoint at 6m.
        // Real limb-slice byte lengths also bound these offsets by usize.
        let (eval_len, product_len, low_product_len, high_offset) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_add(1).unchecked_mul(2),
                split_len.unchecked_mul(2),
                split_len.unchecked_mul(6),
            )
        };
        let ScratchLayout {
            neg_two: at_neg_two,
            neg_one: at_neg_one,
            two: at_two,
            half: at_half,
            eval_a,
            eval_b,
            inner: inner_scratch,
        } = Self::split_scratch::<false>(scratch, product_len, eval_len);

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

        // Admission guarantees four nonempty blocks in both operands, so both
        // endpoint products exist even when a high block contains only zeros.
        // SAFETY: both suffixes' total width is at most the validated full product.
        let high_product_len = unsafe { a3.len().unchecked_add(b3.len()) };
        {
            // SAFETY: the endpoint occupies the initialized full output's prefix.
            let (at_zero, _) = unsafe { dst.split_at_mut_unchecked(low_product_len) };
            Recursive::recursive_mul(at_zero, a0, b0, evaluations.inner, TierCeiling::Toom3);
        }
        {
            // SAFETY: both high parts exist, and their product occupies
            // [6m,|a|+|b|), contained in the validated full output.
            let (at_infinity, _) = unsafe {
                dst.split_at_mut_unchecked(high_offset)
                    .1
                    .split_at_mut_unchecked(high_product_len)
            };
            Recursive::recursive_mul(at_infinity, a3, b3, evaluations.inner, TierCeiling::Toom3);
        }
        Self::clear_destination_outside_products(dst, split_len, product_len, high_product_len);

        // SAFETY: endpoint recursion initialized [0,2m) and [6m,|a|+|b|);
        // multiply_points initialized W(1) at [2m,4m+2), and the preceding
        // gap fills initialized all remaining limbs, including any surplus.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };
        Self::interpolate_destination(
            initialized,
            DestinationInterpolation {
                split_len,
                product_len,
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
        // SAFETY: |a|>3m gives 2|a|>=6m+2, containing W(1)'s [2m,4m+2)
        // buffer and the high square at 6m. These sizes fit usize for real slices.
        let (eval_len, product_len, one_offset, high_offset) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_add(1).unchecked_mul(2),
                split_len.unchecked_mul(2),
                split_len.unchecked_mul(6),
            )
        };
        debug_assert!(
            dst.len() >= a.len().saturating_mul(2),
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
        } = Self::split_scratch::<true>(scratch, product_len, eval_len);

        let (a0, a1, a2, a3) = Self::split_four(a, split_len);

        let kernels = EvaluationKernels {
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
            add_mul: ArchKernels::selected_add_mul_limbs_unchecked(),
        };

        // SAFETY: a product slot contains two complete evaluation widths.
        let (negative_eval, _) = unsafe { at_half.split_at_mut_unchecked(eval_len) };
        let _ = Self::evaluate_pair::<true>(eval, negative_eval, a0, a1, a2, a3, kernels);
        Self::sqr_evaluation(at_neg_two, negative_eval, inner_scratch, kernels.add_mul);
        Self::sqr_evaluation(at_two, eval, inner_scratch, kernels.add_mul);

        Self::evaluate_half_scaled(eval, a0, a1, a2, a3);
        Self::sqr_evaluation(at_half, eval, inner_scratch, kernels.add_mul);

        {
            // SAFETY: four-part admission proved [2m,4m+2) lies in dst;
            // its first m+1 limbs temporarily hold the negative evaluation.
            let at_one = unsafe {
                dst.split_at_mut_unchecked(one_offset)
                    .1
                    .split_at_mut_unchecked(product_len)
                    .0
            };
            // SAFETY: product_len=2*eval_len, so the temporary fits that slot.
            let (negative_one_eval, _) = unsafe { at_one.split_at_mut_unchecked(eval_len) };
            let _ = Self::evaluate_pair::<false>(eval, negative_one_eval, a0, a1, a2, a3, kernels);
            Self::sqr_evaluation(
                at_neg_one,
                negative_one_eval,
                inner_scratch,
                kernels.add_mul,
            );
            Self::sqr_evaluation(at_one, eval, inner_scratch, kernels.add_mul);
        }

        // Four-part admission guarantees a nonempty high endpoint.
        // SAFETY: the doubled high suffix width fits the full 2|a|-limb square.
        let high_product_len = unsafe { a3.len().unchecked_mul(2) };
        {
            // SAFETY: the full m-limb low block squares into the 2m prefix.
            let (at_zero, _) = unsafe { dst.split_at_mut_unchecked(one_offset) };
            Recursive::recursive_sqr(at_zero, a0, inner_scratch, TierCeiling::Toom3);
        }
        {
            // SAFETY: the nonempty high block square spans [6m,2|a|), within dst.
            let (at_infinity, _) = unsafe {
                dst.split_at_mut_unchecked(high_offset)
                    .1
                    .split_at_mut_unchecked(high_product_len)
            };
            Recursive::recursive_sqr(at_infinity, a3, inner_scratch, TierCeiling::Toom3);
        }
        Self::clear_destination_outside_products(dst, split_len, product_len, high_product_len);

        Self::interpolate_destination(
            dst,
            DestinationInterpolation {
                split_len,
                product_len,
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

    fn interpolate_destination(dst: &mut [Limb], values: DestinationInterpolation<'_>) {
        let DestinationInterpolation {
            split_len,
            product_len,
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
        // SAFETY: all products physically span product_len=2m+2>=4 limbs.
        // Their 2m+1 active prefixes fit; admission gives the in-bounds 6m endpoint.
        let (
            active_product_len,
            active_neg_two,
            active_neg_one,
            active_two,
            active_half,
            one_offset,
            high_offset,
        ) = unsafe {
            let active = product_len.unchecked_sub(1);
            (
                active,
                neg_two.split_at_mut_unchecked(active).0,
                neg_one.split_at_mut_unchecked(active).0,
                two.split_at_mut_unchecked(active).0,
                half.split_at_mut_unchecked(active).0,
                split_len.unchecked_mul(2),
                split_len.unchecked_mul(6),
            )
        };
        {
            // SAFETY: the three initialized ranges are [0,2m), [2m,4m+2),
            // and [6m,|a|+|b|). Their intervening gap is 2m-2>=0 because m>=1.
            let (at_zero, at_one, at_infinity) = unsafe {
                let (at_zero, one_and_after) = dst.split_at_mut_unchecked(one_offset);
                let (at_one_storage, after_one) = one_and_after.split_at_mut_unchecked(product_len);
                let at_one = at_one_storage.split_at_mut_unchecked(active_product_len).0;
                let gap = high_offset.unchecked_sub(one_offset.unchecked_add(product_len));
                let infinity_and_after = after_one.split_at_unchecked(gap).1;
                (
                    at_zero,
                    at_one,
                    infinity_and_after.split_at_unchecked(high_product_len).0,
                )
            };
            Self::interpolate_with_endpoints(
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
        Self::reconstruct_around_quadratic(
            dst,
            split_len,
            active_neg_two,
            active_neg_one,
            active_two,
            active_half,
        );
    }

    /// Evaluated guards are below fifteen, so their product fits one limb.
    /// Product children are capped at Toom-4; square children at Toom-3.
    pub fn mul_evaluation(
        dst: &mut [impl LimbOutput],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_product::<15, 1, _>(
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

    fn sqr_evaluation(
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
    fn clear_destination_outside_products(
        dst: &mut [impl LimbOutput],
        split_len: usize,
        product_len: usize,
        high_product_len: usize,
    ) {
        // W(0), W(1), and W(infinity) overwrite their complete destination
        // ranges. W(0) ends exactly at W(1)'s 2m offset. Only the gap after W(1)
        // and the suffix above the infinity endpoint require initialization.
        // SAFETY: four-part admission gives m>=1 and full output length>=6m+2.
        // W(1) ends at 4m+2<=6m, so the intervening gap is 2m-2>=0. The infinity
        // product ends at |a|+|b|<=dst.len(). All splits remain disjoint and in bounds.
        let (gap_after_one, trailing_gap) = unsafe {
            let one_offset = split_len.unchecked_mul(2);
            let high_offset = split_len.unchecked_mul(6);
            let one_and_after = dst.split_at_mut_unchecked(one_offset).1;
            let after_one = one_and_after.split_at_mut_unchecked(product_len).1;
            let gap_len = high_offset.unchecked_sub(one_offset.unchecked_add(product_len));
            let (gap, infinity_and_after) = after_one.split_at_mut_unchecked(gap_len);
            (
                gap,
                infinity_and_after
                    .split_at_mut_unchecked(high_product_len)
                    .1,
            )
        };
        gap_after_one.fill(LimbOutput::from_limb(0));
        trailing_gap.fill(LimbOutput::from_limb(0));
    }

    const fn split_scratch<const SQUARE: bool>(
        scratch: &mut [Limb],
        product_len: usize,
        eval_len: usize,
    ) -> ScratchLayout<'_> {
        let second_eval_len = if SQUARE { 0 } else { eval_len };
        // SAFETY: checked local sizing reserves four 2*eval_len products and
        // one/two eval_len evaluations plus child scratch. Sequential splits
        // preserve initialized, disjoint spans without changing buffer lengths.
        let (neg_two, neg_one, two, half, eval_a, eval_b, inner) = unsafe {
            let (neg_two, after_neg_two) = scratch.split_at_mut_unchecked(product_len);
            let (neg_one, after_neg_one) = after_neg_two.split_at_mut_unchecked(product_len);
            let (two, after_two) = after_neg_one.split_at_mut_unchecked(product_len);
            let (half, after_half) = after_two.split_at_mut_unchecked(product_len);
            let (eval_a, after_eval_a) = after_half.split_at_mut_unchecked(eval_len);
            let (eval_b, inner) = after_eval_a.split_at_mut_unchecked(second_eval_len);
            (neg_two, neg_one, two, half, eval_a, eval_b, inner)
        };

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
}
