//! Evaluation-point orchestration and reciprocal coupling for Toom-8/8.5.

#![expect(
    unsafe_code,
    reason = "Packed point widths, equal guarded evaluations, and complete first writes bound all coupled product spans"
)]

use core::num::NonZeroUsize;

use super::{
    AddMulKernel, ArchKernels, CouplingContext, EvaluationDirection, EvaluationPoint, Limb,
    LimbOutput, PointShift, ProductPair, SharedEval, Toom8, Values,
};

pub struct MulEvaluationBuffers<'buffer> {
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub odd_a: &'buffer mut [Limb],
    pub odd_b: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub fast_paired_add_sub: bool,
    pub add_mul_kernel: AddMulKernel,
}

pub struct SqrEvaluationBuffers<'buffer> {
    pub eval: &'buffer mut [Limb],
    pub odd: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub fast_paired_add_sub: bool,
    pub add_mul_kernel: AddMulKernel,
}

const ONE: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Direct,
    shift: PointShift::Zero,
};
const TWO: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Direct,
    shift: PointShift::One,
};
const FOUR: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Direct,
    shift: PointShift::Two,
};
const EIGHT: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Direct,
    shift: PointShift::Three,
};
const HALF: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Reciprocal,
    shift: PointShift::One,
};
const QUARTER: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Reciprocal,
    shift: PointShift::Two,
};
const EIGHTH: EvaluationPoint = EvaluationPoint {
    direction: EvaluationDirection::Reciprocal,
    shift: PointShift::Three,
};

impl Toom8 {
    pub fn evaluate_and_couple_mul(
        values: &mut Values<'_, impl LimbOutput>,
        temporary: &mut [Limb],
        evaluations: &mut MulEvaluationBuffers<'_>,
        a: &[Limb],
        b: &[Limb],
        context: &CouplingContext<'_>,
    ) {
        Self::evaluate_mul_point(values.one, temporary, evaluations, a, b, context, ONE);
        Self::evaluate_mul_point(values.two, temporary, evaluations, a, b, context, TWO);
        Self::evaluate_mul_point(values.half, temporary, evaluations, a, b, context, HALF);
        // SAFETY: the initialized equal-width two/half packed buffers are
        // disjoint. The kernel writes their sum and reverse difference; final
        // carry/borrow represent only fixed-width sign extension.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.two.as_mut_ptr().cast(),
                values.half.as_mut_ptr().cast(),
                values.two.len(),
            );
        }
        Self::evaluate_mul_point(values.four, temporary, evaluations, a, b, context, FOUR);
        Self::evaluate_mul_point(
            values.quarter,
            temporary,
            evaluations,
            a,
            b,
            context,
            QUARTER,
        );
        // SAFETY: four/quarter are disjoint initialized packed windows of
        // equal checked width; their retained guards bound both signed rows.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.four.as_mut_ptr().cast(),
                values.quarter.as_mut_ptr().cast(),
                values.four.len(),
            );
        }
        Self::evaluate_mul_point(values.eight, temporary, evaluations, a, b, context, EIGHT);
        Self::evaluate_mul_point(values.eighth, temporary, evaluations, a, b, context, EIGHTH);
        // SAFETY: eight/eighth are disjoint initialized packed windows of
        // equal width. The matrix guard retains the signed reverse difference.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.eight.as_mut_ptr().cast(),
                values.eighth.as_mut_ptr().cast(),
                values.eight.len(),
            );
        }
    }

    pub fn evaluate_and_couple_sqr(
        values: &mut Values<'_>,
        temporary: &mut [Limb],
        evaluations: &mut SqrEvaluationBuffers<'_>,
        operand: &[Limb],
        context: &CouplingContext<'_>,
    ) {
        Self::evaluate_sqr_point(values.one, temporary, evaluations, operand, context, ONE);
        Self::evaluate_sqr_point(values.two, temporary, evaluations, operand, context, TWO);
        Self::evaluate_sqr_point(values.half, temporary, evaluations, operand, context, HALF);
        // SAFETY: two/half are equal-width disjoint initialized square-point
        // buffers; the signed row's guard makes carry/borrow sign extension.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.two.as_mut_ptr(),
                values.half.as_mut_ptr(),
                values.two.len(),
            );
        }
        Self::evaluate_sqr_point(values.four, temporary, evaluations, operand, context, FOUR);
        Self::evaluate_sqr_point(
            values.quarter,
            temporary,
            evaluations,
            operand,
            context,
            QUARTER,
        );
        // SAFETY: four/quarter share the checked packed width and are
        // initialized disjoint buffers; their signed sum/difference fit its guard.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.four.as_mut_ptr(),
                values.quarter.as_mut_ptr(),
                values.four.len(),
            );
        }
        Self::evaluate_sqr_point(
            values.eight,
            temporary,
            evaluations,
            operand,
            context,
            EIGHT,
        );
        Self::evaluate_sqr_point(
            values.eighth,
            temporary,
            evaluations,
            operand,
            context,
            EIGHTH,
        );
        // SAFETY: eight/eighth are equal-width initialized disjoint buffers.
        // The finite interpolation matrix bounds both resulting signed rows.
        unsafe {
            let _ = ArchKernels::add_reverse_sub_limbs_unchecked(
                values.eight.as_mut_ptr(),
                values.eighth.as_mut_ptr(),
                values.eight.len(),
            );
        }
    }
    #[inline]
    fn evaluate_mul_point(
        packed: &mut [impl LimbOutput],
        temporary: &mut [Limb],
        evaluations: &mut MulEvaluationBuffers<'_>,
        a: &[Limb],
        b: &[Limb],
        context: &CouplingContext<'_>,
        evaluation_point: EvaluationPoint,
    ) {
        // SAFETY: packed has 3m+g limbs and temporary the same width. Removing
        // its m-limb prefix yields 2m+g<=temporary.len(), the exact point width.
        let negative = unsafe {
            temporary
                .split_at_mut_unchecked(packed.len().unchecked_sub(context.split_len.get()))
                .0
        };
        let sign = Self::evaluate_mul_pair(
            packed,
            negative,
            evaluations,
            a,
            b,
            context.split_len,
            evaluation_point,
        );
        match evaluation_point.direction {
            EvaluationDirection::Direct => {
                Self::couple_direct(packed, negative, sign, context, evaluation_point.shift);
            }
            EvaluationDirection::Reciprocal => {
                Self::couple_reciprocal(packed, negative, sign, context, evaluation_point.shift);
            }
        }
    }

    #[inline]
    fn evaluate_sqr_point(
        packed: &mut [Limb],
        temporary: &mut [Limb],
        evaluations: &mut SqrEvaluationBuffers<'_>,
        operand: &[Limb],
        context: &CouplingContext<'_>,
        evaluation_point: EvaluationPoint,
    ) {
        // SAFETY: the square layout gives packed/temporary 3m+g limbs, so its
        // 2m+g-limb point-product prefix fits the temporary without subtraction overflow.
        let negative = unsafe {
            temporary
                .split_at_mut_unchecked(packed.len().unchecked_sub(context.split_len.get()))
                .0
        };
        Self::evaluate_sqr_pair(
            packed,
            negative,
            evaluations,
            operand,
            context.split_len,
            evaluation_point,
        );
        match evaluation_point.direction {
            EvaluationDirection::Direct => {
                Self::couple_direct(packed, negative, false, context, evaluation_point.shift);
            }
            EvaluationDirection::Reciprocal => {
                Self::couple_reciprocal(packed, negative, false, context, evaluation_point.shift);
            }
        }
    }

    fn evaluate_mul_pair<Output: LimbOutput>(
        packed: &mut [Output],
        negative: &mut [Limb],
        buffers: &mut MulEvaluationBuffers<'_>,
        a: &[Limb],
        b: &[Limb],
        split_width: NonZeroUsize,
        point: EvaluationPoint,
    ) -> bool {
        let split_len = split_width.get();
        debug_assert_eq!(
            packed.len().saturating_sub(split_len),
            negative.len(),
            "packed point product has the wrong width"
        );
        // SAFETY: the driver supplies packed.len()=3m+g and negative.len()=2m+g,
        // so the m-limb prefix leaves exactly one disjoint point-product window.
        let positive = unsafe { packed.split_at_mut_unchecked(split_len).1 };
        // SAFETY: negative is initialized scratch, and each recursive product
        // stores only initialized limb values without deinitializing any slot.
        let negative_output = unsafe { Output::from_initialized_mut(negative) };
        let pair = ProductPair {
            positive,
            negative: negative_output,
        };

        let negative_a = Self::evaluate_even_odd(
            buffers.eval_a,
            buffers.odd_a,
            a,
            split_width,
            point,
            buffers.add_mul_kernel,
        );
        let negative_b = Self::evaluate_even_odd(
            buffers.eval_b,
            buffers.odd_b,
            b,
            split_width,
            point,
            buffers.add_mul_kernel,
        );
        let negative_product_is_negative = negative_a ^ negative_b;
        let fast_paired = buffers.fast_paired_add_sub;
        if fast_paired {
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval_a,
                buffers.odd_a,
                negative_a,
            );
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval_b,
                buffers.odd_b,
                negative_b,
            );
        } else {
            // SAFETY: each Toom-8 even/odd pair is allocated with identical
            // guarded widths by the evaluation scratch layout.
            unsafe {
                SharedEval::add_part(buffers.eval_a, buffers.odd_a);
            }
            // SAFETY: the second operand pair has the same exact-width layout.
            unsafe {
                SharedEval::add_part(buffers.eval_b, buffers.odd_b);
            }
        }
        // For W(-x)=-N, route (N,P) into (packed,temporary); otherwise route
        // (P,N). Coupling then forms E in the packed high window in either case,
        // eliminating the full-window relocation and high-window zero fill.
        let (positive_dst, magnitude_dst) = if negative_product_is_negative {
            (pair.negative, pair.positive)
        } else {
            (pair.positive, pair.negative)
        };
        Self::mul_evaluation(
            positive_dst,
            buffers.eval_a,
            buffers.eval_b,
            buffers.scratch,
            split_len,
            buffers.add_mul_kernel,
        );

        if fast_paired {
            Self::mul_evaluation(
                magnitude_dst,
                buffers.odd_a,
                buffers.odd_b,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        } else {
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval_a,
                buffers.odd_a,
                negative_a,
            );
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval_b,
                buffers.odd_b,
                negative_b,
            );
            Self::mul_evaluation(
                magnitude_dst,
                buffers.eval_a,
                buffers.eval_b,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        }
        negative_product_is_negative
    }

    fn evaluate_sqr_pair(
        packed: &mut [Limb],
        negative: &mut [Limb],
        buffers: &mut SqrEvaluationBuffers<'_>,
        operand: &[Limb],
        split_width: NonZeroUsize,
        point: EvaluationPoint,
    ) {
        let split_len = split_width.get();
        debug_assert_eq!(
            packed.len().saturating_sub(split_len),
            negative.len(),
            "packed point square has the wrong width"
        );
        // SAFETY: the packed square table is 3m+g limbs and its m-limb prefix
        // leaves the exact 2m+g-limb point-product width, disjoint from negative.
        let positive = unsafe { packed.split_at_mut_unchecked(split_len).1 };
        let pair = ProductPair { positive, negative };

        let negative_evaluation = Self::evaluate_even_odd(
            buffers.eval,
            buffers.odd,
            operand,
            split_width,
            point,
            buffers.add_mul_kernel,
        );
        let fast_paired = buffers.fast_paired_add_sub;
        if fast_paired {
            SharedEval::apply_sum_and_absolute_difference(
                buffers.eval,
                buffers.odd,
                negative_evaluation,
            );
        } else {
            // SAFETY: `eval` and `odd` are the equal-width guarded buffers of
            // one Toom-8 conjugate evaluation pair.
            unsafe {
                SharedEval::add_part(buffers.eval, buffers.odd);
            }
        }
        Self::sqr_evaluation(
            pair.positive,
            buffers.eval,
            buffers.scratch,
            split_len,
            buffers.add_mul_kernel,
        );
        if fast_paired {
            Self::sqr_evaluation(
                pair.negative,
                buffers.odd,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        } else {
            SharedEval::overwrite_sum_with_absolute_difference(
                buffers.eval,
                buffers.odd,
                negative_evaluation,
            );
            Self::sqr_evaluation(
                pair.negative,
                buffers.eval,
                buffers.scratch,
                split_len,
                buffers.add_mul_kernel,
            );
        }
    }
}
