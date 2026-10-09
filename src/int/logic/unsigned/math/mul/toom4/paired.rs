//! Paired positive and negative evaluation for balanced Toom-Cook 4.

#![expect(
    unsafe_code,
    reason = "Equal guarded widths and nonempty four-part admission bound paired evaluations and temporary product slots"
)]

use core::ptr::copy_nonoverlapping;

use super::{AddMulKernel, ArchKernels, Limb, LimbOutput, SharedEval, Toom4};

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
        dst: &mut [impl LimbOutput],
        dimensions: PointDimensions,
        parts_a: OperandParts<'_>,
        parts_b: OperandParts<'_>,
        products: &mut MiddleProducts<'_>,
        evaluations: &mut EvaluationBuffers<'_>,
    ) -> (bool, bool) {
        let neg_one_negative = {
            // SAFETY: each product slot spans two evaluation widths; its
            // negative evaluations die before the negative-two product is written.
            let (neg_one_a, neg_one_b) = unsafe {
                products
                    .neg_two
                    .split_at_mut_unchecked(dimensions.evaluation)
            };
            let sign_a = Self::evaluate_pair::<false>(
                evaluations.a,
                neg_one_a,
                parts_a.zero,
                parts_a.one,
                parts_a.two,
                parts_a.three,
                evaluations.kernels,
            );
            let sign_b = Self::evaluate_pair::<false>(
                evaluations.b,
                neg_one_b,
                parts_b.zero,
                parts_b.one,
                parts_b.two,
                parts_b.three,
                evaluations.kernels,
            );
            sign_a ^ sign_b
        };
        let signs = Self::multiply_paired_points(
            dst,
            dimensions,
            parts_a,
            parts_b,
            products,
            evaluations,
            neg_one_negative,
        );

        // Select the half-point evaluator once for both operands.
        if evaluations.kernels.fast_paired_add_sub {
            Self::evaluate_half_scaled_with_kernel(
                evaluations.a,
                [parts_a.zero, parts_a.one, parts_a.two, parts_a.three],
                evaluations.kernels.add_mul,
            );
            Self::evaluate_half_scaled_with_kernel(
                evaluations.b,
                [parts_b.zero, parts_b.one, parts_b.two, parts_b.three],
                evaluations.kernels.add_mul,
            );
        } else {
            Self::evaluate_half_scaled(
                evaluations.a,
                parts_a.zero,
                parts_a.one,
                parts_a.two,
                parts_a.three,
            );
            Self::evaluate_half_scaled(
                evaluations.b,
                parts_b.zero,
                parts_b.one,
                parts_b.two,
                parts_b.three,
            );
        }
        Self::mul_evaluation(
            products.half,
            evaluations.a,
            evaluations.b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );
        signs
    }
    fn multiply_paired_points(
        dst: &mut [impl LimbOutput],
        dimensions: PointDimensions,
        parts_a: OperandParts<'_>,
        parts_b: OperandParts<'_>,
        products: &mut MiddleProducts<'_>,
        evaluations: &mut EvaluationBuffers<'_>,
        neg_one_negative: bool,
    ) -> (bool, bool) {
        // SAFETY: admitted operands exceed 3m each, so dst contains [2m,4m+2).
        // PointDimensions carries product=2m+2 and evaluation=m+1 from the driver.
        let at_one = unsafe {
            dst.split_at_mut_unchecked(dimensions.split.unchecked_mul(2))
                .1
                .split_at_mut_unchecked(dimensions.product)
                .0
        };
        Self::mul_evaluation(
            at_one,
            evaluations.a,
            evaluations.b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );
        // SAFETY: the initialized temporary slot spans two evaluation widths.
        let (negative_one_a, negative_one_b) =
            unsafe { products.neg_two.split_at_unchecked(dimensions.evaluation) };
        Self::mul_evaluation(
            products.neg_one,
            negative_one_a,
            negative_one_b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );

        // SAFETY: the unused half-product slot also spans two evaluation widths.
        let (negative_two_operand_a, negative_two_operand_b) =
            unsafe { products.half.split_at_mut_unchecked(dimensions.evaluation) };
        let sign_a = Self::evaluate_pair::<true>(
            evaluations.a,
            negative_two_operand_a,
            parts_a.zero,
            parts_a.one,
            parts_a.two,
            parts_a.three,
            evaluations.kernels,
        );
        let sign_b = Self::evaluate_pair::<true>(
            evaluations.b,
            negative_two_operand_b,
            parts_b.zero,
            parts_b.one,
            parts_b.two,
            parts_b.three,
            evaluations.kernels,
        );
        Self::mul_evaluation(
            products.two,
            evaluations.a,
            evaluations.b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );
        // SAFETY: both initialized negative evaluations retain their exact widths
        // until their product consumes them; the half-product is written afterward.
        let (stored_negative_two_a, stored_negative_two_b) =
            unsafe { products.half.split_at_unchecked(dimensions.evaluation) };
        Self::mul_evaluation(
            products.neg_two,
            stored_negative_two_a,
            stored_negative_two_b,
            evaluations.inner,
            evaluations.kernels.add_mul,
        );
        (sign_a ^ sign_b, neg_one_negative)
    }

    /// Evaluate one operand at `+1/-1` or `+2/-2` in two fused limb passes.
    ///
    /// The first pass forms the even and odd polynomial parts. The second forms
    /// their sum and absolute difference, returning the sign of the negative-point
    /// value. The driver admits three complete low blocks and one nonempty
    /// high block of at most the split width; both outputs retain one guard.
    pub fn evaluate_pair<const AT_TWO: bool>(
        positive: &mut [Limb],
        negative: &mut [Limb],
        part0: &[Limb],
        part1: &[Limb],
        part2: &[Limb],
        part3: &[Limb],
        kernels: EvaluationKernels,
    ) -> bool {
        debug_assert!(positive.len() > 1, "evaluation includes a body and guard");
        debug_assert_eq!(positive.len(), negative.len(), "paired widths match");
        // SAFETY: split_scratch supplies split_len + 1 limbs to both outputs.
        let (positive_guard, positive_body) =
            unsafe { positive.split_last_mut().unwrap_unchecked() };
        // SAFETY: negative has the same nonzero guarded width as positive.
        let (negative_guard, negative_body) =
            unsafe { negative.split_last_mut().unwrap_unchecked() };
        debug_assert!(
            part0.len() == positive_body.len()
                && part1.len() == positive_body.len()
                && part2.len() == positive_body.len()
                && part3.len() <= positive_body.len(),
            "the admitted four-part split has three full low blocks"
        );

        if kernels.fast_paired_add_sub && !AT_TWO && part3.len() == positive_body.len() {
            // SAFETY: the admitted split gives three full low parts, and this
            // branch establishes a full high part. Both outputs are disjoint
            // initialized buffers of that nonzero width.
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
            return SharedEval::sum_and_absolute_difference(positive, negative);
        }

        if kernels.fast_paired_add_sub && AT_TWO {
            // SAFETY: admission provides full m-limb constant/linear parts
            // and equal-width initialized scratch bodies. The operands and
            // the two writable bodies occupy disjoint spans.
            unsafe {
                copy_nonoverlapping(part0.as_ptr(), positive_body.as_mut_ptr(), part0.len());
                copy_nonoverlapping(part1.as_ptr(), negative_body.as_mut_ptr(), part1.len());
            }
            *positive_guard = 0;
            *negative_guard = 0;
            SharedEval::add_mul_word_with_kernel_in_place(positive, part2, 4, kernels.add_mul);
            SharedEval::add_mul_word_with_kernel_in_place(negative, part3, 4, kernels.add_mul);
            // SAFETY: `negative` is a valid writable evaluation buffer and the
            // shift by one is below every supported limb width. Before the shift,
            // a1+4*a3 < 5*B^m; therefore 2*a1+8*a3 < 10*B^m fits its guard.
            let carry =
                unsafe { ArchKernels::lshift_unchecked(negative.as_mut_ptr(), negative.len(), 1) };
            debug_assert_eq!(carry, 0, "odd evaluation exceeded its guard limb");
            return SharedEval::sum_and_absolute_difference(positive, negative);
        }

        let prefix_len = part3.len();
        // SAFETY: the admitted low blocks and both destination bodies span m
        // limbs; the high block has prefix_len<=m. The mutable bodies are disjoint.
        let (
            (even_prefix, even_suffix),
            (odd_prefix, odd_suffix),
            (part0_prefix, part0_suffix),
            (part1_prefix, part1_suffix),
            (part2_prefix, part2_suffix),
        ) = unsafe {
            (
                positive_body.split_at_mut_unchecked(prefix_len),
                negative_body.split_at_mut_unchecked(prefix_len),
                part0.split_at_unchecked(prefix_len),
                part1.split_at_unchecked(prefix_len),
                part2.split_at_unchecked(prefix_len),
            )
        };
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

        SharedEval::sum_and_absolute_difference(positive, negative)
    }
}

/// Unit-weight recurrence: `left+right+carry<=2B-1` preserves carry in {0,1}.
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
    // SAFETY: all flags are zero or one; the positive sum of their weights
    // is at most 2+1+1+1=5, below Limb::MAX on every supported target.
    *carry = unsafe {
        Limb::from(overflow_a)
            .unchecked_mul(2)
            .unchecked_add(Limb::from(overflow_b))
            .unchecked_add(Limb::from(overflow_c))
            .unchecked_add(Limb::from(overflow_d))
    };
    complete
}

fn evaluate_odd_at_two(part1: Limb, part3: Limb, carry: &mut Limb) -> Limb {
    let (part1_twice, overflow_a) = part1.overflowing_add(part1);
    let (part3_twice, overflow_b) = part3.overflowing_add(part3);
    let (part3_four, overflow_c) = part3_twice.overflowing_add(part3_twice);
    let (part3_eight, overflow_d) = part3_four.overflowing_add(part3_four);
    let (sum, overflow_e) = part1_twice.overflowing_add(part3_eight);
    let (complete, overflow_f) = sum.overflowing_add(*carry);
    // SAFETY: flag weights sum to 1+4+2+1+1+1=10, below Limb::MAX even
    // at 16 bits. Inductively this exact 2*a1+8*a3 recurrence has carry<=9.
    *carry = unsafe {
        Limb::from(overflow_a)
            .unchecked_add(Limb::from(overflow_b).unchecked_mul(4))
            .unchecked_add(Limb::from(overflow_c).unchecked_mul(2))
            .unchecked_add(Limb::from(overflow_d))
            .unchecked_add(Limb::from(overflow_e))
            .unchecked_add(Limb::from(overflow_f))
    };
    complete
}
