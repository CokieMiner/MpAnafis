//! Paired positive and negative evaluation for balanced Toom-Cook 4.

use super::{AddMulKernel, ArchKernels, Limb, SharedEval, Toom4};

/// Four polynomial parts of one Toom-4 operand.
#[derive(Clone, Copy)]
pub struct OperandParts<'value> {
    pub zero: &'value [Limb],
    pub one: &'value [Limb],
    pub two: &'value [Limb],
    pub three: &'value [Limb],
}

/// Product buffers retained for interpolation.
pub struct MiddleProducts<'buffer> {
    pub neg_two: &'buffer mut [Limb],
    pub neg_one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
}

/// Reusable operand and recursive-work buffers for point products.
pub struct EvaluationBuffers<'buffer> {
    pub a: &'buffer mut [Limb],
    pub b: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
    pub kernels: EvaluationKernels,
}

#[derive(Clone, Copy)]
pub struct EvaluationKernels {
    pub fast_paired_add_sub: bool,
    pub add_mul: AddMulKernel,
}

/// Fixed widths and destination position for one Toom-4 level.
#[derive(Clone, Copy)]
pub struct PointDimensions {
    pub split: usize,
    pub evaluation: usize,
    pub product: usize,
}

impl Toom4 {
    /// Evaluate and multiply all five non-endpoint Toom-4 points.
    pub fn multiply_points(
        dst: &mut [Limb],
        dimensions: PointDimensions,
        parts_a: OperandParts<'_>,
        parts_b: OperandParts<'_>,
        products: &mut MiddleProducts<'_>,
        evaluations: &mut EvaluationBuffers<'_>,
    ) -> (bool, bool) {
        let paired_signs = {
            let (neg_one_a, neg_one_b) = products.neg_two.split_at_mut(dimensions.evaluation);
            let sign_a = Self::evaluate_positive_and_negative_balanced::<false>(
                evaluations.a,
                neg_one_a,
                parts_a.zero,
                parts_a.one,
                parts_a.two,
                parts_a.three,
                evaluations.kernels,
            );
            let sign_b = Self::evaluate_positive_and_negative_balanced::<false>(
                evaluations.b,
                neg_one_b,
                parts_b.zero,
                parts_b.one,
                parts_b.two,
                parts_b.three,
                evaluations.kernels,
            );
            sign_a.zip(sign_b)
        };
        let signs = if let Some((neg_one_a, neg_one_b)) = paired_signs {
            multiply_paired_points(
                dst,
                dimensions,
                parts_a,
                parts_b,
                products,
                evaluations,
                neg_one_a ^ neg_one_b,
            )
        } else {
            multiply_general_points(dst, dimensions, parts_a, parts_b, products, evaluations)
        };

        evaluate_half_scaled_selected(evaluations.a, parts_a, evaluations.kernels);
        evaluate_half_scaled_selected(evaluations.b, parts_b, evaluations.kernels);
        Self::recursive_mul_evaluation(
            products.half,
            evaluations.a,
            evaluations.b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );
        signs
    }
}

fn evaluate_half_scaled_selected(
    dst: &mut [Limb],
    parts: OperandParts<'_>,
    kernels: EvaluationKernels,
) {
    let accelerated = kernels.fast_paired_add_sub
        && Toom4::evaluate_half_scaled_with_kernel(
            dst,
            [parts.zero, parts.one, parts.two, parts.three],
            kernels.add_mul,
        );
    if !accelerated {
        Toom4::evaluate_half_scaled(dst, parts.zero, parts.one, parts.two, parts.three);
    }
}

fn multiply_paired_points(
    dst: &mut [Limb],
    dimensions: PointDimensions,
    parts_a: OperandParts<'_>,
    parts_b: OperandParts<'_>,
    products: &mut MiddleProducts<'_>,
    evaluations: &mut EvaluationBuffers<'_>,
    neg_one_negative: bool,
) -> (bool, bool) {
    let one_offset = dimensions.split.wrapping_mul(2);
    let (_, one_and_after) = dst.split_at_mut(one_offset);
    let (at_one, _) = one_and_after.split_at_mut(dimensions.product);
    Toom4::recursive_mul_evaluation(
        at_one,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    let (negative_one_a, negative_one_b) = products.neg_two.split_at(dimensions.evaluation);
    Toom4::recursive_mul_evaluation(
        products.neg_one,
        negative_one_a,
        negative_one_b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );

    let (negative_two_operand_a, negative_two_operand_b) =
        products.half.split_at_mut(dimensions.evaluation);
    let sign_a = Toom4::evaluate_positive_and_negative_balanced::<true>(
        evaluations.a,
        negative_two_operand_a,
        parts_a.zero,
        parts_a.one,
        parts_a.two,
        parts_a.three,
        evaluations.kernels,
    );
    let sign_b = Toom4::evaluate_positive_and_negative_balanced::<true>(
        evaluations.b,
        negative_two_operand_b,
        parts_b.zero,
        parts_b.one,
        parts_b.two,
        parts_b.three,
        evaluations.kernels,
    );
    debug_assert!(
        sign_a.is_some() && sign_b.is_some(),
        "paired +/-1 support must imply paired +/-2 support"
    );
    Toom4::recursive_mul_evaluation(
        products.two,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    let (stored_negative_two_a, stored_negative_two_b) =
        products.half.split_at(dimensions.evaluation);
    Toom4::recursive_mul_evaluation(
        products.neg_two,
        stored_negative_two_a,
        stored_negative_two_b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    // SAFETY: paired +/-1 implies paired +/-2 by construction.
    let negative_two_a = unsafe {
        // SAFETY: sign_a is Some when paired +/-1 exists.
        sign_a.unwrap_unchecked()
    };
    // SAFETY: paired +/-1 implies paired +/-2 by construction.
    let negative_two_b = unsafe {
        // SAFETY: sign_b is Some when paired +/-1 exists.
        sign_b.unwrap_unchecked()
    };
    (negative_two_a ^ negative_two_b, neg_one_negative)
}

fn multiply_general_points(
    dst: &mut [Limb],
    dimensions: PointDimensions,
    parts_a: OperandParts<'_>,
    parts_b: OperandParts<'_>,
    products: &mut MiddleProducts<'_>,
    evaluations: &mut EvaluationBuffers<'_>,
) -> (bool, bool) {
    for (evaluation, parts) in [
        (&mut *evaluations.a, parts_a),
        (&mut *evaluations.b, parts_b),
    ] {
        Toom4::evaluate_positive(evaluation, parts.zero, parts.one, parts.two, parts.three, 1);
    }
    Toom4::recursive_mul_evaluation(
        products.two,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    let (negative_two_temp, _) = products.neg_two.split_at_mut(dimensions.evaluation);
    let sign_two_a = Toom4::evaluate_negative_magnitude(
        evaluations.a,
        negative_two_temp,
        parts_a.zero,
        parts_a.one,
        parts_a.two,
        parts_a.three,
        1,
    );
    let sign_two_b = Toom4::evaluate_negative_magnitude(
        evaluations.b,
        negative_two_temp,
        parts_b.zero,
        parts_b.one,
        parts_b.two,
        parts_b.three,
        1,
    );
    Toom4::recursive_mul_evaluation(
        products.neg_two,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    for (evaluation, parts) in [
        (&mut *evaluations.a, parts_a),
        (&mut *evaluations.b, parts_b),
    ] {
        Toom4::evaluate_positive(evaluation, parts.zero, parts.one, parts.two, parts.three, 0);
    }
    let one_offset = dimensions.split.wrapping_mul(2);
    let (_, one_and_after) = dst.split_at_mut(one_offset);
    let (at_one, _) = one_and_after.split_at_mut(dimensions.product);
    Toom4::recursive_mul_evaluation(
        at_one,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    let (negative_one_temp, _) = products.neg_one.split_at_mut(dimensions.evaluation);
    let sign_one_a = Toom4::evaluate_negative_magnitude(
        evaluations.a,
        negative_one_temp,
        parts_a.zero,
        parts_a.one,
        parts_a.two,
        parts_a.three,
        0,
    );
    let sign_one_b = Toom4::evaluate_negative_magnitude(
        evaluations.b,
        negative_one_temp,
        parts_b.zero,
        parts_b.one,
        parts_b.two,
        parts_b.three,
        0,
    );
    Toom4::recursive_mul_evaluation(
        products.neg_one,
        evaluations.a,
        evaluations.b,
        evaluations.inner,
        evaluations.kernels.add_mul,
    );
    (sign_two_a ^ sign_two_b, sign_one_a ^ sign_one_b)
}

impl Toom4 {
    /// Evaluate one operand at `+1/-1` or `+2/-2` in two fused limb passes.
    ///
    /// The first pass forms the even and odd polynomial parts. The second forms
    /// their sum and absolute difference, returning the sign of the negative-point
    /// value. `None` leaves callers on the general partial-part evaluator.
    pub fn evaluate_positive_and_negative_balanced<const AT_TWO: bool>(
        positive: &mut [Limb],
        negative: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
        kernels: EvaluationKernels,
    ) -> Option<bool> {
        let (positive_guard, positive_body) = positive.split_last_mut()?;
        let (negative_guard, negative_body) = negative.split_last_mut()?;
        if positive_body.is_empty()
            || part0.len() != positive_body.len()
            || part1.len() != positive_body.len()
            || part2.len() != positive_body.len()
            || part3.len() > positive_body.len()
            || negative_body.len() != positive_body.len()
        {
            return None;
        }

        if kernels.fast_paired_add_sub && !AT_TWO && part3.len() == positive_body.len() {
            // SAFETY: validation above proves all four parts and both destination
            // bodies have the same nonzero width and the destinations are disjoint.
            *positive_guard = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    positive_body.as_mut_ptr(),
                    part0.as_ptr(),
                    part2.as_ptr(),
                    positive_body.len(),
                )
            };
            // SAFETY: the same equal-width proof covers the disjoint odd sum.
            *negative_guard = unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    negative_body.as_mut_ptr(),
                    part1.as_ptr(),
                    part3.as_ptr(),
                    negative_body.len(),
                )
            };
            return Some(SharedEval::sum_and_absolute_difference(positive, negative));
        }

        if kernels.fast_paired_add_sub && AT_TWO {
            positive_body.copy_from_slice(part0);
            *positive_guard = 0;
            negative_body.copy_from_slice(part1);
            *negative_guard = 0;
            SharedEval::add_mul_word_with_kernel_in_place(positive, part2, 4, kernels.add_mul);
            SharedEval::add_mul_word_with_kernel_in_place(negative, part3, 4, kernels.add_mul);
            // SAFETY: `negative` is a valid writable evaluation buffer and the
            // shift by one is below every supported limb width. Before the shift,
            // a1+4*a3 < 5*B^m; therefore 2*a1+8*a3 < 10*B^m fits its guard.
            let carry =
                unsafe { ArchKernels::lshift_unchecked(negative.as_mut_ptr(), negative.len(), 1) };
            debug_assert_eq!(carry, 0, "odd evaluation exceeded its guard limb");
            return Some(SharedEval::sum_and_absolute_difference(positive, negative));
        }

        let prefix_len = part3.len();
        let (even_prefix, even_suffix) = positive_body.split_at_mut(prefix_len);
        let (odd_prefix, odd_suffix) = negative_body.split_at_mut(prefix_len);
        let (part0_prefix, part0_suffix) = part0.split_at(prefix_len);
        let (part1_prefix, part1_suffix) = part1.split_at(prefix_len);
        let (part2_prefix, part2_suffix) = part2.split_at(prefix_len);
        let mut even_carry = 0;
        let mut odd_carry = 0;
        for (((((even_limb, odd_limb), part0_limb), part1_limb), part2_limb), part3_limb) in
            even_prefix
                .iter_mut()
                .zip(odd_prefix)
                .zip(part0_prefix)
                .zip(part1_prefix)
                .zip(part2_prefix)
                .zip(part3)
        {
            if AT_TWO {
                *even_limb = evaluate_even_at_two(*part0_limb, *part2_limb, &mut even_carry);
                *odd_limb = evaluate_odd_at_two(*part1_limb, *part3_limb, &mut odd_carry);
            } else {
                *even_limb = evaluate_at_one(*part0_limb, *part2_limb, &mut even_carry);
                *odd_limb = evaluate_at_one(*part1_limb, *part3_limb, &mut odd_carry);
            }
        }
        for ((((even_limb, odd_limb), part0_limb), part1_limb), part2_limb) in even_suffix
            .iter_mut()
            .zip(odd_suffix)
            .zip(part0_suffix)
            .zip(part1_suffix)
            .zip(part2_suffix)
        {
            if AT_TWO {
                *even_limb = evaluate_even_at_two(*part0_limb, *part2_limb, &mut even_carry);
                *odd_limb = evaluate_odd_at_two(*part1_limb, Limb::MIN, &mut odd_carry);
            } else {
                *even_limb = evaluate_at_one(*part0_limb, *part2_limb, &mut even_carry);
                *odd_limb = evaluate_at_one(*part1_limb, Limb::MIN, &mut odd_carry);
            }
        }
        *positive_guard = even_carry;
        *negative_guard = odd_carry;

        Some(SharedEval::sum_and_absolute_difference(positive, negative))
    }
}

fn evaluate_at_one(left: Limb, right: Limb, carry: &mut Limb) -> Limb {
    let (sum, overflow_a) = left.overflowing_add(right);
    let (complete, overflow_b) = sum.overflowing_add(*carry);
    *carry = Limb::from(overflow_a | overflow_b);
    complete
}

fn evaluate_even_at_two(part0: Limb, part2: Limb, carry: &mut Limb) -> Limb {
    let (twice, overflow_a) = part2.overflowing_add(part2);
    let (four_times, overflow_b) = twice.overflowing_add(twice);
    let (sum, overflow_c) = part0.overflowing_add(four_times);
    let (complete, overflow_d) = sum.overflowing_add(*carry);
    *carry = Limb::from(overflow_a)
        .wrapping_mul(2)
        .wrapping_add(Limb::from(overflow_b))
        .wrapping_add(Limb::from(overflow_c))
        .wrapping_add(Limb::from(overflow_d));
    complete
}

fn evaluate_odd_at_two(part1: Limb, part3: Limb, carry: &mut Limb) -> Limb {
    let (part1_twice, overflow_a) = part1.overflowing_add(part1);
    let (part3_twice, overflow_b) = part3.overflowing_add(part3);
    let (part3_four, overflow_c) = part3_twice.overflowing_add(part3_twice);
    let (part3_eight, overflow_d) = part3_four.overflowing_add(part3_four);
    let (sum, overflow_e) = part1_twice.overflowing_add(part3_eight);
    let (complete, overflow_f) = sum.overflowing_add(*carry);
    *carry = Limb::from(overflow_a)
        .wrapping_add(Limb::from(overflow_b).wrapping_mul(4))
        .wrapping_add(Limb::from(overflow_c).wrapping_mul(2))
        .wrapping_add(Limb::from(overflow_d))
        .wrapping_add(Limb::from(overflow_e))
        .wrapping_add(Limb::from(overflow_f));
    complete
}
