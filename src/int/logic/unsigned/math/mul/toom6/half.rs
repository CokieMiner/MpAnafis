//! Unbalanced seven-by-six Toom-Cook 6.5 multiplication.

use core::cmp::{max, min};

use super::{
    ArchKernels, Limb, MulEvaluationBuffers, Multiplication, Parts, Recursive, ScratchLayout,
    TierCeiling, Toom6, Values,
};

impl Toom6 {
    /// Scratch length required by the seven-by-six Toom-6.5 split.
    pub fn half_scratch_len(len_a: usize, len_b: usize) -> usize {
        let larger = max(len_a, len_b);
        let smaller = min(len_a, len_b);
        let split_len = half_split_len(larger, smaller);
        let eval_len = split_len.wrapping_add(1);
        let high_large_len = larger.saturating_sub(split_len.wrapping_mul(6));
        let high_small_len = smaller.saturating_sub(split_len.wrapping_mul(5));
        let evaluation_inner = if split_len.is_power_of_two() {
            let plan = Multiplication::select_plan(split_len, split_len, TierCeiling::Toom4);
            Multiplication::scratch_len(plan, split_len, split_len)
        } else {
            let plan = Multiplication::select_plan(eval_len, eval_len, TierCeiling::Toom4);
            Multiplication::scratch_len(plan, eval_len, eval_len)
        };
        let plan_low = Multiplication::select_plan(split_len, split_len, TierCeiling::Toom4);
        let low_inner = Multiplication::scratch_len(plan_low, split_len, split_len);
        let plan_high =
            Multiplication::select_plan(high_large_len, high_small_len, TierCeiling::Toom4);
        let high_inner = Multiplication::scratch_len(plan_high, high_large_len, high_small_len);
        Self::local_scratch_len(split_len, max(evaluation_inner, max(low_inner, high_inner)))
    }

    /// Multiply operands represented by seven and six radix-`B^m` chunks.
    pub fn half_mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let split_len = half_split_len(larger.len(), smaller.len());
        let eval_len = split_len.wrapping_add(1);
        let value_len = split_len.wrapping_mul(2).wrapping_add(3);
        debug_assert!(
            dst.len() >= a.len().wrapping_add(b.len()),
            "Toom-6.5 multiplication output is shorter than the full product"
        );
        debug_assert!(
            scratch.len() >= Self::half_scratch_len(a.len(), b.len()),
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
        } = Self::split_scratch(scratch, value_len, eval_len);
        let large_parts = Self::split_seven(larger, split_len);
        let small_parts = Self::split_six(smaller, split_len);
        debug_assert!(
            !large_parts.sextic.is_empty() && !small_parts.quintic.is_empty(),
            "Toom-6.5 requires nonempty degree-six and degree-five endpoints"
        );

        let zero_product_len = large_parts
            .constant
            .len()
            .wrapping_add(small_parts.constant.len());
        let (zero_product, higher_terms) = dst.split_at_mut(zero_product_len);
        higher_terms.fill(0);
        Recursive::recursive_mul(
            zero_product,
            large_parts.constant,
            small_parts.constant,
            inner,
            TierCeiling::Toom4,
        );

        let infinity_offset = split_len.wrapping_mul(11);
        let infinity_len = large_parts
            .sextic
            .len()
            .wrapping_add(small_parts.quintic.len());
        let (_, mutable_infinity_and_tail) = dst.split_at_mut(infinity_offset);
        let (infinity_product, _) = mutable_infinity_and_tail.split_at_mut(infinity_len);
        Recursive::recursive_mul(
            infinity_product,
            large_parts.sextic,
            small_parts.quintic,
            inner,
            TierCeiling::Toom4,
        );

        let mut evaluations = MulEvaluationBuffers {
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            scratch: inner,
            add_mul_kernel: ArchKernels::selected_add_mul_limbs_unchecked(),
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
        };
        let zero_len = split_len.wrapping_mul(2);
        let (before_infinity, infinity_and_tail) = dst.split_at(infinity_offset);
        let (zero, _) = before_infinity.split_at(zero_len);
        let (infinity, _) = infinity_and_tail.split_at(infinity_len);

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
                zero,
                infinity,
                split_len,
            },
        );
        Self::interpolate_and_reconstruct(dst, split_len, values);
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
        0,
    );
    Toom6::couple_direct_half(
        values.one, temporary, sign_one, zero, infinity, split_len, 0,
    );
    let sign_two = Toom6::evaluate_mul_pair(
        values.two,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        1,
    );
    Toom6::couple_direct_half(
        values.two, temporary, sign_two, zero, infinity, split_len, 1,
    );
    let sign_four = Toom6::evaluate_mul_pair(
        values.four,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        2,
    );
    Toom6::couple_direct_half(
        values.four,
        temporary,
        sign_four,
        zero,
        infinity,
        split_len,
        2,
    );
    let sign_half = Toom6::evaluate_reciprocal_mul_pair(
        values.half,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        1,
    );
    Toom6::couple_reciprocal_half(
        values.half,
        temporary,
        sign_half,
        zero,
        infinity,
        split_len,
        1,
    );
    let sign_quarter = Toom6::evaluate_reciprocal_mul_pair(
        values.quarter,
        temporary,
        evaluations,
        large_parts,
        small_parts,
        2,
    );
    Toom6::couple_reciprocal_half(
        values.quarter,
        temporary,
        sign_quarter,
        zero,
        infinity,
        split_len,
        2,
    );
}

fn half_split_len(larger: usize, smaller: usize) -> usize {
    max(larger.div_ceil(7), smaller.div_ceil(6))
}
