//! Unbalanced seven-by-six Toom-Cook 6.5 multiplication.

#![expect(
    unsafe_code,
    reason = "Seven-by-six admission proves nonempty endpoints, guarded evaluations, and disjoint initialized reconstruction spans"
)]

use core::cmp::{max, min};

use super::{
    ArchKernels, EvaluationDirection, Limb, LimbOutput, MulEvaluationBuffers, Multiplication,
    Parts, PointShift, Recursive, ScratchLayout, TierCeiling, Toom6, Values,
};

impl Toom6 {
    /// Scratch length required by the seven-by-six Toom-6.5 split.
    pub fn mul_half_scratch_len(len_a: usize, len_b: usize) -> usize {
        let larger = max(len_a, len_b);
        let smaller = min(len_a, len_b);
        let split_len = max(larger.div_ceil(7), smaller.div_ceil(6));
        // SAFETY: this sizing branch follows toom6_half_suitable, which proves
        // larger>6m and smaller>5m using saturating products. An overflow would
        // make either comparison false. Both offsets and subtractions thus fit,
        // and m<larger/6 leaves room for its single evaluation guard.
        let (high_large_len, high_small_len, eval_len) = unsafe {
            (
                larger.unchecked_sub(split_len.unchecked_mul(6)),
                smaller.unchecked_sub(split_len.unchecked_mul(5)),
                split_len.unchecked_add(1),
            )
        };
        let plan_low = Multiplication::select_plan(split_len, split_len, TierCeiling::Toom4);
        let low_inner = Multiplication::scratch_len(plan_low, split_len, split_len);
        let evaluation_inner = if split_len.is_power_of_two() {
            low_inner
        } else {
            let plan = Multiplication::select_plan(eval_len, eval_len, TierCeiling::Toom4);
            Multiplication::scratch_len(plan, eval_len, eval_len)
        };
        let plan_high =
            Multiplication::select_plan(high_large_len, high_small_len, TierCeiling::Toom4);
        let high_inner = Multiplication::scratch_len(plan_high, high_large_len, high_small_len);
        Self::local_scratch_len::<false>(
            split_len,
            max(evaluation_inner, max(low_inner, high_inner)),
            false,
        )
    }

    /// Multiply operands represented by seven and six radix-`B^m` chunks.
    pub fn mul_half(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let split_len = max(larger.len().div_ceil(7), smaller.len().div_ceil(6));
        // SAFETY: nonempty endpoints prove 11m<|a|+|b|<=dst.len(), bounding
        // m+1 and 2m+2 for every actual admitted seven-by-six product.
        let (eval_len, value_len) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_mul(2).unchecked_add(2),
            )
        };
        debug_assert!(
            scratch.len() >= Self::mul_half_scratch_len(a.len(), b.len()),
            "Toom-6.5 multiplication scratch buffer is undersized"
        );

        let ScratchLayout {
            one,
            two,
            four,
            half,
            quarter,
            temporary,
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            inner,
        } = Self::split_scratch::<false>(scratch, value_len, eval_len, false);
        let large_parts = Self::split_seven(larger, split_len);
        let small_parts = Self::split_six(smaller, split_len);

        // SAFETY: admission gives larger.len()>6m and smaller.len()>5m. Both
        // full m-limb constants and nonempty endpoint suffixes exist. Thus 2m,
        // 11m and the endpoint width fit their complete product destination,
        // whose real byte span bounds all three on 16/32/64-bit targets.
        let (zero_product_len, infinity_offset, infinity_len) = unsafe {
            (
                split_len.unchecked_mul(2),
                split_len.unchecked_mul(11),
                large_parts
                    .sextic
                    .len()
                    .unchecked_add(small_parts.quintic.len()),
            )
        };
        // SAFETY: the endpoints occupy [0,2m) and [11m,|a|+|b|), proven
        // disjoint and contained in dst. Sequential splits preserve their
        // initialized storage and separate the two reconstruction gaps.
        let (zero_product, middle_gap, infinity_product, trailing_gap) = unsafe {
            let (before_infinity, infinity_and_tail) = dst.split_at_mut_unchecked(infinity_offset);
            let (zero_product, middle_gap) =
                before_infinity.split_at_mut_unchecked(zero_product_len);
            let (infinity_product, trailing_gap) =
                infinity_and_tail.split_at_mut_unchecked(infinity_len);
            (zero_product, middle_gap, infinity_product, trailing_gap)
        };
        // Both recursive products overwrite their complete endpoint spans.
        // Only the disjoint reconstruction gaps require an initial zero fill.
        middle_gap.fill(LimbOutput::from_limb(0));
        trailing_gap.fill(LimbOutput::from_limb(0));
        Recursive::recursive_mul(
            zero_product,
            large_parts.constant,
            small_parts.constant,
            inner,
            TierCeiling::Toom4,
        );

        Recursive::recursive_mul(
            infinity_product,
            large_parts.sextic,
            small_parts.quintic,
            inner,
            TierCeiling::Toom4,
        );

        // SAFETY: both endpoint recursions wrote their exact disjoint spans.
        let (zero_value, infinity_value) = unsafe {
            (
                LimbOutput::assume_init(zero_product),
                LimbOutput::assume_init(infinity_product),
            )
        };

        let mut evaluations = MulEvaluationBuffers {
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            scratch: inner,
            add_mul_kernel: ArchKernels::selected_add_mul_limbs_unchecked(),
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
        };
        let mut values = Values {
            one,
            two,
            four,
            half,
            quarter,
        };
        evaluate_and_couple(
            &mut values,
            temporary,
            &mut evaluations,
            large_parts,
            small_parts,
            &HalfEndpoints {
                zero: zero_value,
                infinity: infinity_value,
                split_len,
            },
        );
        // SAFETY: the endpoint writers and the disjoint middle/suffix fills
        // cover every destination limb; endpoint borrows ended after evaluation.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };
        Self::interpolate_and_reconstruct(initialized, split_len, values);
    }
}

struct HalfEndpoints<'value> {
    zero: &'value [Limb],
    infinity: &'value [Limb],
    split_len: usize,
}

fn evaluate_and_couple(
    values: &mut Values<'_>,
    temporary: &mut [Limb],
    evaluations: &mut MulEvaluationBuffers<'_>,
    large_parts: Parts<'_>,
    small_parts: Parts<'_>,
    endpoints: &HalfEndpoints<'_>,
) {
    let zero = endpoints.zero;
    let infinity = endpoints.infinity;
    let split_len = endpoints.split_len;
    let sign_one = Toom6::evaluate_mul_pair(
        values.one,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        EvaluationDirection::Direct,
        PointShift::Zero,
    );
    Toom6::couple_direct_half(
        values.one,
        temporary,
        sign_one,
        zero,
        infinity,
        split_len,
        PointShift::Zero,
    );
    let sign_two = Toom6::evaluate_mul_pair(
        values.two,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        EvaluationDirection::Direct,
        PointShift::One,
    );
    Toom6::couple_direct_half(
        values.two,
        temporary,
        sign_two,
        zero,
        infinity,
        split_len,
        PointShift::One,
    );
    let sign_four = Toom6::evaluate_mul_pair(
        values.four,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        EvaluationDirection::Direct,
        PointShift::Two,
    );
    Toom6::couple_direct_half(
        values.four,
        temporary,
        sign_four,
        zero,
        infinity,
        split_len,
        PointShift::Two,
    );
    let sign_half = Toom6::evaluate_mul_pair(
        values.half,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        EvaluationDirection::Reciprocal,
        PointShift::One,
    );
    Toom6::couple_reciprocal_half(
        values.half,
        temporary,
        sign_half,
        zero,
        infinity,
        split_len,
        PointShift::One,
    );
    let sign_quarter = Toom6::evaluate_mul_pair(
        values.quarter,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        EvaluationDirection::Reciprocal,
        PointShift::Two,
    );
    Toom6::couple_reciprocal_half(
        values.quarter,
        temporary,
        sign_quarter,
        zero,
        infinity,
        split_len,
        PointShift::Two,
    );
}
