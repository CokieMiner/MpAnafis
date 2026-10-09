//! Conjugate-pair products, and the five-point schedules built from them.
//!
//! Each point `k` is multiplied together with its conjugate `-k`, because both
//! come from the same even and odd accumulators. `A(k)B(k)` is nonnegative, while
//! `A(-k)B(-k)` is signed; the pair is stored as that product and the magnitude of
//! the other, with the sign returned to the interpolator.

#![expect(
    unsafe_code,
    reason = "Equal guarded evaluation widths and complete packed product layouts prove temporary span and initialization bounds"
)]

use super::{
    AddMulKernel, EvaluationDirection, Limb, LimbOutput, Parts, PointShift, ProductPair, Recursive,
    SharedEval, TierCeiling, Toom6, Values,
};

pub struct MulEvaluationBuffers<'buffer> {
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub odd_a: &'buffer mut [Limb],
    pub odd_b: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub add_mul_kernel: AddMulKernel,
    pub fast_paired_add_sub: bool,
}

pub struct SqrEvaluationBuffers<'buffer> {
    pub eval: &'buffer mut [Limb],
    pub odd: &'buffer mut [Limb],
    pub scratch: &'buffer mut [Limb],
    pub add_mul_kernel: AddMulKernel,
    pub fast_paired_add_sub: bool,
}

impl Toom6 {
    /// Runs the five-point multiplication schedule, coupling each pair as it lands.
    pub fn evaluate_mul_points(
        values: &mut Values<'_, impl LimbOutput>,
        temporary: &mut [Limb],
        evaluations: &mut MulEvaluationBuffers<'_>,
        parts_a: Parts<'_>,
        parts_b: Parts<'_>,
        zero: &[Limb],
        split_len: usize,
    ) {
        let sign_one = Self::evaluate_mul_pair(
            values.one,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            EvaluationDirection::Direct,
            PointShift::Zero,
        );
        Self::couple_direct(
            values.one,
            temporary,
            sign_one,
            zero,
            split_len,
            PointShift::Zero,
        );
        let sign_two = Self::evaluate_mul_pair(
            values.two,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            EvaluationDirection::Direct,
            PointShift::One,
        );
        Self::couple_direct(
            values.two,
            temporary,
            sign_two,
            zero,
            split_len,
            PointShift::One,
        );
        let sign_four = Self::evaluate_mul_pair(
            values.four,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            EvaluationDirection::Direct,
            PointShift::Two,
        );
        Self::couple_direct(
            values.four,
            temporary,
            sign_four,
            zero,
            split_len,
            PointShift::Two,
        );
        let sign_half = Self::evaluate_mul_pair(
            values.half,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            EvaluationDirection::Reciprocal,
            PointShift::One,
        );
        Self::couple_reciprocal(
            values.half,
            temporary,
            sign_half,
            zero,
            split_len,
            PointShift::One,
        );
        let sign_quarter = Self::evaluate_mul_pair(
            values.quarter,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            EvaluationDirection::Reciprocal,
            PointShift::Two,
        );
        Self::couple_reciprocal(
            values.quarter,
            temporary,
            sign_quarter,
            zero,
            split_len,
            PointShift::Two,
        );
    }

    /// The squaring counterpart of [`Self::evaluate_mul_points`].
    ///
    /// A square's negative-point value is never negative, so every coupling here is
    /// given a `false` sign.
    pub fn evaluate_sqr_points(
        values: &mut Values<'_>,
        temporary: &mut [Limb],
        evaluations: &mut SqrEvaluationBuffers<'_>,
        parts: Parts<'_>,
        zero: &[Limb],
        split_len: usize,
    ) {
        evaluate_sqr_pair_at(
            &mut split_pair(values.one, temporary),
            evaluations,
            parts,
            EvaluationDirection::Direct,
            PointShift::Zero,
        );
        Self::couple_direct(
            values.one,
            temporary,
            false,
            zero,
            split_len,
            PointShift::Zero,
        );
        evaluate_sqr_pair_at(
            &mut split_pair(values.two, temporary),
            evaluations,
            parts,
            EvaluationDirection::Direct,
            PointShift::One,
        );
        Self::couple_direct(
            values.two,
            temporary,
            false,
            zero,
            split_len,
            PointShift::One,
        );
        evaluate_sqr_pair_at(
            &mut split_pair(values.four, temporary),
            evaluations,
            parts,
            EvaluationDirection::Direct,
            PointShift::Two,
        );
        Self::couple_direct(
            values.four,
            temporary,
            false,
            zero,
            split_len,
            PointShift::Two,
        );
        evaluate_sqr_pair_at(
            &mut split_pair(values.half, temporary),
            evaluations,
            parts,
            EvaluationDirection::Reciprocal,
            PointShift::One,
        );
        Self::couple_reciprocal(
            values.half,
            temporary,
            false,
            zero,
            split_len,
            PointShift::One,
        );
        evaluate_sqr_pair_at(
            &mut split_pair(values.quarter, temporary),
            evaluations,
            parts,
            EvaluationDirection::Reciprocal,
            PointShift::Two,
        );
        Self::couple_reciprocal(
            values.quarter,
            temporary,
            false,
            zero,
            split_len,
            PointShift::Two,
        );
    }
    /// Evaluate and multiply a direct or reciprocal conjugate pair, returning
    /// whether the product at its negative point is negative.
    pub fn evaluate_mul_pair(
        packed: &mut [impl LimbOutput],
        negative: &mut [Limb],
        buffers: &mut MulEvaluationBuffers<'_>,
        parts_a: Parts<'_>,
        parts_b: Parts<'_>,
        direction: EvaluationDirection,
        shift: PointShift,
    ) -> bool {
        let pair = split_pair(packed, negative);
        let kernel = buffers.add_mul_kernel;
        let negative_a = Self::evaluate_even_odd(
            buffers.eval_a,
            buffers.odd_a,
            parts_a,
            direction,
            shift,
            kernel,
        );
        let negative_b = Self::evaluate_even_odd(
            buffers.eval_b,
            buffers.odd_b,
            parts_b,
            direction,
            shift,
            kernel,
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
            // SAFETY: each even/odd pair is allocated with the same guarded
            // evaluation width by the Toom-6 scratch layout.
            unsafe {
                SharedEval::add_part(buffers.eval_a, buffers.odd_a);
            }
            // SAFETY: the second operand pair has the same exact-width layout.
            unsafe {
                SharedEval::add_part(buffers.eval_b, buffers.odd_b);
            }
        }

        // P=A(k)B(k) and N=|A(-k)B(-k)|. When the signed negative-point
        // product is -N, placing N in the packed window and P in the temporary
        // window lets coupling form E=(P-N)/2 directly in its final B^m-shifted
        // location. The nonnegative case retains the conventional P,N layout.
        // Thus neither sign needs to relocate a full 2m+2-limb table afterward.
        let (positive_dst, magnitude_dst) = if negative_product_is_negative {
            (&mut *pair.negative, &mut *pair.positive)
        } else {
            (&mut *pair.positive, &mut *pair.negative)
        };
        mul_evaluation(
            positive_dst,
            buffers.eval_a,
            buffers.eval_b,
            buffers.scratch,
            kernel,
        );

        if fast_paired {
            mul_evaluation(
                magnitude_dst,
                buffers.odd_a,
                buffers.odd_b,
                buffers.scratch,
                kernel,
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
            mul_evaluation(
                magnitude_dst,
                buffers.eval_a,
                buffers.eval_b,
                buffers.scratch,
                kernel,
            );
        }
        negative_product_is_negative
    }
}

fn evaluate_sqr_pair_at(
    pair: &mut ProductPair<'_>,
    buffers: &mut SqrEvaluationBuffers<'_>,
    parts: Parts<'_>,
    direction: EvaluationDirection,
    shift: PointShift,
) {
    let kernel = buffers.add_mul_kernel;
    let negative =
        Toom6::evaluate_even_odd(buffers.eval, buffers.odd, parts, direction, shift, kernel);
    let fast_paired = buffers.fast_paired_add_sub;
    if fast_paired {
        SharedEval::apply_sum_and_absolute_difference(buffers.eval, buffers.odd, negative);
    } else {
        // SAFETY: `eval` and `odd` are the equal-width guarded buffers of one
        // conjugate evaluation pair.
        unsafe {
            SharedEval::add_part(buffers.eval, buffers.odd);
        }
    }
    sqr_evaluation(pair.positive, buffers.eval, buffers.scratch, kernel);
    if fast_paired {
        sqr_evaluation(pair.negative, buffers.odd, buffers.scratch, kernel);
    } else {
        SharedEval::overwrite_sum_with_absolute_difference(buffers.eval, buffers.odd, negative);
        sqr_evaluation(pair.negative, buffers.eval, buffers.scratch, kernel);
    }
}

/// Views the packed window and the temporary as one positive/negative pair.
fn split_pair<'buffer, Output: LimbOutput>(
    packed: &'buffer mut [Output],
    negative: &'buffer mut [Limb],
) -> ProductPair<'buffer, Output> {
    // SAFETY: the driver supplies packed.len()=3m+2 and negative.len()=2m+2.
    // Their difference m is nonnegative and bounded by packed.len(); the tail
    // is already exactly one product width, so no second split is required.
    let positive = unsafe {
        packed
            .split_at_mut_unchecked(packed.len().unchecked_sub(negative.len()))
            .1
    };
    // SAFETY: negative is initialized scratch. Both product writers retain
    // initialized contents throughout; no path deinitializes this limb storage.
    let negative_output = unsafe { Output::from_initialized_mut(negative) };
    ProductPair {
        positive,
        negative: negative_output,
    }
}

fn mul_evaluation(
    dst: &mut [impl LimbOutput],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
    kernel: AddMulKernel,
) {
    // Power-of-two bodies retain the fixed-width child specializations.
    // The degree-six guard product requires two limbs on 16-bit targets.
    // SAFETY: every point evaluation spans m+1 limbs with m>=1.
    let low_len = unsafe { a.len().unchecked_sub(1) };
    if !low_len.is_power_of_two() {
        Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
        return;
    }
    Recursive::guarded_evaluation_product::<5_462, 2, _>(
        dst,
        a,
        b,
        scratch,
        kernel,
        |product, low_a, low_b, s| {
            Recursive::recursive_mul(product, low_a, low_b, s, TierCeiling::Toom4);
        },
    );
}

fn sqr_evaluation(dst: &mut [Limb], value: &[Limb], scratch: &mut [Limb], kernel: AddMulKernel) {
    // SAFETY: every square evaluation spans m+1 limbs with m>=1.
    let low_len = unsafe { value.len().unchecked_sub(1) };
    if !low_len.is_power_of_two() {
        Recursive::recursive_sqr(dst, value, scratch, TierCeiling::Toom4);
        return;
    }
    Recursive::guarded_evaluation_square::<5_462, 2>(
        dst,
        value,
        scratch,
        kernel,
        |square, low, s| {
            Recursive::recursive_sqr(square, low, s, TierCeiling::Toom4);
        },
    );
}
