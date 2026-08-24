//! Conjugate-pair products, and the five-point schedules built from them.
//!
//! Each point `k` is multiplied together with its conjugate `-k`, because both
//! come from the same even and odd accumulators. `A(k)B(k)` is nonnegative, while
//! `A(-k)B(-k)` is signed; the pair is stored as that product and the magnitude of
//! the other, with the sign returned to the interpolator.

use super::{
    AddMulKernel, EvaluationDirection, Limb, Parts, ProductPair, Recursive, SharedEval,
    TierCeiling, Toom6, Values,
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
    /// Multiplies the conjugate pair at `2^point_shift`, returning whether the
    /// negative-point product is negative.
    pub fn evaluate_mul_pair(
        packed: &mut [Limb],
        negative: &mut [Limb],
        buffers: &mut MulEvaluationBuffers<'_>,
        parts_a: Parts<'_>,
        parts_b: Parts<'_>,
        point_shift: u32,
    ) -> bool {
        evaluate_mul_pair_at(
            &mut split_pair(packed, negative),
            buffers,
            parts_a,
            parts_b,
            EvaluationDirection::Direct,
            point_shift,
        )
    }

    /// The reciprocal-point counterpart of [`Self::evaluate_mul_pair`], at
    /// `2^-denominator_shift`.
    pub fn evaluate_reciprocal_mul_pair(
        packed: &mut [Limb],
        negative: &mut [Limb],
        buffers: &mut MulEvaluationBuffers<'_>,
        parts_a: Parts<'_>,
        parts_b: Parts<'_>,
        denominator_shift: u32,
    ) -> bool {
        evaluate_mul_pair_at(
            &mut split_pair(packed, negative),
            buffers,
            parts_a,
            parts_b,
            EvaluationDirection::Reciprocal,
            denominator_shift,
        )
    }

    /// Runs the five-point multiplication schedule, coupling each pair as it lands.
    pub fn evaluate_mul_points(
        values: &mut Values<'_>,
        temporary: &mut [Limb],
        evaluations: &mut MulEvaluationBuffers<'_>,
        parts_a: Parts<'_>,
        parts_b: Parts<'_>,
        zero: &[Limb],
        split_len: usize,
    ) {
        let sign_one =
            Self::evaluate_mul_pair(values.one, temporary, evaluations, parts_a, parts_b, 0);
        Self::couple_direct(values.one, temporary, sign_one, zero, split_len, 0);
        let sign_two =
            Self::evaluate_mul_pair(values.two, temporary, evaluations, parts_a, parts_b, 1);
        Self::couple_direct(values.two, temporary, sign_two, zero, split_len, 1);
        let sign_four =
            Self::evaluate_mul_pair(values.four, temporary, evaluations, parts_a, parts_b, 2);
        Self::couple_direct(values.four, temporary, sign_four, zero, split_len, 2);
        let sign_half = Self::evaluate_reciprocal_mul_pair(
            values.half,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            1,
        );
        Self::couple_reciprocal(values.half, temporary, sign_half, zero, split_len, 1);
        let sign_quarter = Self::evaluate_reciprocal_mul_pair(
            values.quarter,
            temporary,
            evaluations,
            parts_a,
            parts_b,
            2,
        );
        Self::couple_reciprocal(values.quarter, temporary, sign_quarter, zero, split_len, 2);
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
        evaluate_sqr_pair(values.one, temporary, evaluations, parts, 0);
        Self::couple_direct(values.one, temporary, false, zero, split_len, 0);
        evaluate_sqr_pair(values.two, temporary, evaluations, parts, 1);
        Self::couple_direct(values.two, temporary, false, zero, split_len, 1);
        evaluate_sqr_pair(values.four, temporary, evaluations, parts, 2);
        Self::couple_direct(values.four, temporary, false, zero, split_len, 2);
        evaluate_reciprocal_sqr_pair(values.half, temporary, evaluations, parts, 1);
        Self::couple_reciprocal(values.half, temporary, false, zero, split_len, 1);
        evaluate_reciprocal_sqr_pair(values.quarter, temporary, evaluations, parts, 2);
        Self::couple_reciprocal(values.quarter, temporary, false, zero, split_len, 2);
    }
}

/// Views the packed window and the temporary as one positive/negative pair.
const fn split_pair<'buffer>(
    packed: &'buffer mut [Limb],
    negative: &'buffer mut [Limb],
) -> ProductPair<'buffer> {
    let split_len = packed.len().wrapping_sub(negative.len());
    let (_, positive_and_guard) = packed.split_at_mut(split_len);
    let (positive, _) = positive_and_guard.split_at_mut(negative.len());
    ProductPair { positive, negative }
}

fn evaluate_sqr_pair(
    packed: &mut [Limb],
    negative: &mut [Limb],
    buffers: &mut SqrEvaluationBuffers<'_>,
    parts: Parts<'_>,
    point_shift: u32,
) {
    evaluate_sqr_pair_at(
        &mut split_pair(packed, negative),
        buffers,
        parts,
        EvaluationDirection::Direct,
        point_shift,
    );
}

fn evaluate_reciprocal_sqr_pair(
    packed: &mut [Limb],
    negative: &mut [Limb],
    buffers: &mut SqrEvaluationBuffers<'_>,
    parts: Parts<'_>,
    denominator_shift: u32,
) {
    evaluate_sqr_pair_at(
        &mut split_pair(packed, negative),
        buffers,
        parts,
        EvaluationDirection::Reciprocal,
        denominator_shift,
    );
}

#[allow(
    unsafe_code,
    reason = "Paired Toom-6 even and odd evaluation buffers have identical guarded widths"
)]
fn evaluate_mul_pair_at(
    pair: &mut ProductPair<'_>,
    buffers: &mut MulEvaluationBuffers<'_>,
    parts_a: Parts<'_>,
    parts_b: Parts<'_>,
    direction: EvaluationDirection,
    shift: u32,
) -> bool {
    let kernel = buffers.add_mul_kernel;
    let negative_a = Toom6::evaluate_even_odd(
        buffers.eval_a,
        buffers.odd_a,
        parts_a,
        direction,
        shift,
        kernel,
    );
    let negative_b = Toom6::evaluate_even_odd(
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
        SharedEval::apply_sum_and_absolute_difference(buffers.eval_a, buffers.odd_a, negative_a);
        SharedEval::apply_sum_and_absolute_difference(buffers.eval_b, buffers.odd_b, negative_b);
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
    // Thus neither sign needs to relocate a full 2m+3-limb table afterward.
    let (positive_dst, magnitude_dst) = if negative_product_is_negative {
        (&mut *pair.negative, &mut *pair.positive)
    } else {
        (&mut *pair.positive, &mut *pair.negative)
    };
    multiply_evaluation(
        positive_dst,
        buffers.eval_a,
        buffers.eval_b,
        buffers.scratch,
        kernel,
    );

    if fast_paired {
        multiply_evaluation(
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
        multiply_evaluation(
            magnitude_dst,
            buffers.eval_a,
            buffers.eval_b,
            buffers.scratch,
            kernel,
        );
    }
    negative_product_is_negative
}

#[allow(
    unsafe_code,
    reason = "Paired Toom-6 square evaluation buffers have identical guarded widths"
)]
fn evaluate_sqr_pair_at(
    pair: &mut ProductPair<'_>,
    buffers: &mut SqrEvaluationBuffers<'_>,
    parts: Parts<'_>,
    direction: EvaluationDirection,
    shift: u32,
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
    square_evaluation(pair.positive, buffers.eval, buffers.scratch, kernel);
    if fast_paired {
        square_evaluation(pair.negative, buffers.odd, buffers.scratch, kernel);
    } else {
        SharedEval::overwrite_sum_with_absolute_difference(buffers.eval, buffers.odd, negative);
        square_evaluation(pair.negative, buffers.eval, buffers.scratch, kernel);
    }
}

fn multiply_evaluation(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
    kernel: AddMulKernel,
) {
    // The guard split is only worth taking when the low half is a power of two,
    // which is where the child tier's fixed-width specializations live. The
    // degree-six guard bound of 5462 does not square into one limb on a 16-bit
    // target, so the guard product spans two limbs.
    if !a.len().wrapping_sub(1).is_power_of_two() {
        Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
        return;
    }
    Recursive::guarded_evaluation_product::<5_462, 2>(
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

fn square_evaluation(dst: &mut [Limb], value: &[Limb], scratch: &mut [Limb], kernel: AddMulKernel) {
    if !value.len().wrapping_sub(1).is_power_of_two() {
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
