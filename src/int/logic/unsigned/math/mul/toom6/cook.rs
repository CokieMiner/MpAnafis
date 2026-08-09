//! The Toom-Cook 6 and 6.5 drivers: split, evaluate, interpolate, reconstruct.
use core::cmp::max;

use super::{
    ArchKernels, Limb, MulEvaluationBuffers, MulShape, Multiplication, Recursive,
    SqrEvaluationBuffers, TierCeiling, Values, Widths,
};

/// Namespace for the six-way and six-and-a-half-way Toom-Cook tiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom6;

impl Toom6 {
    /// Multiply two balanced limb slices with a six-way Toom-Cook split.
    #[allow(
        clippy::too_many_lines,
        reason = "The six-point driver keeps its scratch layout and alternating reconstruction in one cohesive sequence"
    )]
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 6 || b.len() < 6 {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
            return;
        }
        let Some(shape) = Widths::new(a.len(), b.len()).toom6_shape() else {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
            return;
        };
        if matches!(shape, MulShape::Half) {
            Self::half_mul(dst, a, b, scratch);
            return;
        }

        let split_len = max(a.len(), b.len()).div_ceil(6);
        let eval_len = split_len.wrapping_add(1);
        let value_len = split_len.wrapping_mul(2).wrapping_add(3);
        let fit_packed_len = split_len.wrapping_mul(3).wrapping_add(3);
        let place_alternating =
            split_len >= 3 && split_len.wrapping_mul(7).wrapping_add(fit_packed_len) <= dst.len();
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
        let parts_a = Self::split_six(a, split_len);
        let parts_b = Self::split_six(b, split_len);

        // Preserve c0 directly in the output. The recursive product initializes
        // its complete endpoint, while the two placed interpolation buffers also
        // overwrite their complete ranges, so clear only the remaining canvas.
        let zero_product_len = parts_a.constant.len().wrapping_add(parts_b.constant.len());
        clear_mul_destination(dst, zero_product_len, split_len, place_alternating);
        let (zero_product, _) = dst.split_at_mut(zero_product_len);
        Recursive::recursive_mul(
            zero_product,
            parts_a.constant,
            parts_b.constant,
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
        let packed_len = value_len.wrapping_add(split_len);
        let two_offset = split_len.wrapping_mul(3);
        let half_offset = split_len.wrapping_mul(7);
        if place_alternating {
            // The packed coefficient pairs produced from the x=2 and x=1/2 tables
            // finish at radix shifts 3m and 7m. Those ranges are disjoint for
            // m>=3, so retain both directly in their final output positions and
            // keep the constant endpoint in the low prefix.
            {
                let (before_two, two_and_after) = dst.split_at_mut(two_offset);
                let (placed_two, after_two) = two_and_after.split_at_mut(packed_len);
                let gap_len = half_offset.wrapping_sub(two_offset.wrapping_add(packed_len));
                let (_, half_and_after) = after_two.split_at_mut(gap_len);
                let (placed_half, _) = half_and_after.split_at_mut(packed_len);
                let (zero, _) = before_two.split_at(zero_len);
                let mut values = Values {
                    one: &mut *one,
                    two: placed_two,
                    four: &mut *four,
                    half: placed_half,
                    quarter: &mut *quarter,
                };
                #[rustfmt::skip]
                Self::evaluate_mul_points(
                    &mut values,
                    temporary,
                    &mut evaluations,
                    parts_a,
                    parts_b,
                    zero,
                    split_len,
                );
                Self::interpolate_values(values);
            }
            Self::reconstruct_alternating(dst, split_len, four, one, quarter);
        } else {
            let (zero, _) = dst.split_at(zero_len);
            let mut values = Values {
                one,
                two,
                four,
                half,
                quarter,
            };
            #[rustfmt::skip]
            Self::evaluate_mul_points(
                &mut values,
                temporary,
                &mut evaluations,
                parts_a,
                parts_b,
                zero,
                split_len,
            );
            Self::interpolate_and_reconstruct(dst, split_len, values);
        }
    }

    /// Square a limb slice with a six-way Toom-Cook split.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 6 {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom4);
            return;
        }

        let split_len = a.len().div_ceil(6);
        let eval_len = split_len.wrapping_add(1);
        let value_len = split_len.wrapping_mul(2).wrapping_add(3);
        debug_assert!(
            dst.len() >= a.len().wrapping_mul(2),
            "Toom-6 squaring output is shorter than the full square"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom6_sqr_scratch_len(a.len()),
            "Toom-6 squaring scratch buffer is undersized"
        );

        let ScratchLayout {
            one,
            two,
            four,
            half,
            quarter,
            temporary,
            eval_a,
            odd_a,
            inner,
            ..
        } = Self::split_scratch(scratch, value_len, eval_len);
        let parts = Self::split_six(a, split_len);

        // Preserve c0 directly in the output and clear every higher coefficient
        // position before reconstruction starts adding the interpolated terms.
        let zero_product_len = parts.constant.len().wrapping_mul(2);
        let (zero_product, higher_terms) = dst.split_at_mut(zero_product_len);
        higher_terms.fill(0);
        Recursive::recursive_sqr(zero_product, parts.constant, inner, TierCeiling::Toom4);
        let mut evaluations = SqrEvaluationBuffers {
            eval: eval_a,
            odd: odd_a,
            scratch: inner,
            add_mul_kernel: ArchKernels::selected_add_mul_limbs_unchecked(),
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
        };
        let zero_len = split_len.wrapping_mul(2);
        let (zero, _) = dst.split_at(zero_len);

        let mut values = Values {
            one,
            two,
            four,
            half,
            quarter,
        };
        Self::evaluate_sqr_points(
            &mut values,
            temporary,
            &mut evaluations,
            parts,
            zero,
            split_len,
        );
        Self::interpolate_and_reconstruct(dst, split_len, values);
    }
}

pub struct ProductPair<'buffer> {
    pub positive: &'buffer mut [Limb],
    pub negative: &'buffer mut [Limb],
}

// ── Scratch layout and destination clearing ───────────────────────────────────

pub struct ScratchLayout<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
    pub temporary: &'buffer mut [Limb],
    pub eval_a: &'buffer mut [Limb],
    pub eval_b: &'buffer mut [Limb],
    pub odd_a: &'buffer mut [Limb],
    pub odd_b: &'buffer mut [Limb],
    pub inner: &'buffer mut [Limb],
}

impl Toom6 {
    pub const fn split_scratch(
        scratch: &mut [Limb],
        value_len: usize,
        eval_len: usize,
    ) -> ScratchLayout<'_> {
        let split_len = eval_len.wrapping_sub(1);
        let packed_len = value_len.wrapping_add(split_len);
        let (one, after_one) = scratch.split_at_mut(packed_len);
        let (two, after_two) = after_one.split_at_mut(packed_len);
        let (four, after_four) = after_two.split_at_mut(packed_len);
        let (half, after_half) = after_four.split_at_mut(packed_len);
        let (quarter, after_quarter) = after_half.split_at_mut(packed_len);
        let (temporary, after_temporary) = after_quarter.split_at_mut(value_len);
        let (eval_a, after_eval_a) = after_temporary.split_at_mut(eval_len);
        let (eval_b, after_eval_b) = after_eval_a.split_at_mut(eval_len);
        let (odd_a, after_odd_a) = after_eval_b.split_at_mut(eval_len);
        let (odd_b, inner) = after_odd_a.split_at_mut(eval_len);

        ScratchLayout {
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
        }
    }

    pub const fn local_scratch_len(split_len: usize, inner_space: usize) -> usize {
        let eval_len = split_len.wrapping_add(1);
        let value_len = split_len.wrapping_mul(2).wrapping_add(3);
        let packed_len = value_len.wrapping_add(split_len);
        packed_len
            .wrapping_mul(5)
            .wrapping_add(value_len)
            .wrapping_add(eval_len.wrapping_mul(4))
            .wrapping_add(inner_space)
    }
}

fn clear_mul_destination(
    dst: &mut [Limb],
    zero_product_len: usize,
    split_len: usize,
    place_alternating: bool,
) {
    if !place_alternating {
        let (_, reconstruction_canvas) = dst.split_at_mut(zero_product_len);
        reconstruction_canvas.fill(0);
        return;
    }

    let packed_len = split_len.wrapping_mul(3).wrapping_add(3);
    let two_offset = split_len.wrapping_mul(3);
    let half_offset = split_len.wrapping_mul(7);
    let (before_two, two_and_after) = dst.split_at_mut(two_offset);
    let (_, clear_before_two) = before_two.split_at_mut(zero_product_len);
    clear_before_two.fill(0);
    let (_, after_two) = two_and_after.split_at_mut(packed_len);
    let gap_len = half_offset.wrapping_sub(two_offset.wrapping_add(packed_len));
    let (gap, half_and_after) = after_two.split_at_mut(gap_len);
    gap.fill(0);
    let (_, after_half) = half_and_after.split_at_mut(packed_len);
    after_half.fill(0);
}
