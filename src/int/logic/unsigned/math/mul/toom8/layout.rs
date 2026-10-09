//! Shape selection and scratch layout for Toom-8 and Toom-8.5.

#![expect(
    unsafe_code,
    reason = "Shape admission, checked scratch totals, and validated destination intervals bound every consecutive split and endpoint"
)]

use core::{
    cmp::{max, min},
    num::NonZeroUsize,
};

use super::{
    AddMulKernel, ChildDemand, EVALUATION_GUARD_BITS, INTERPOLATION_GUARD_LIMBS, LIMB_BITS, Limb,
    LimbOutput, MulEvaluationBuffers, MulShape, Multiplication, Recursive, SqrEvaluationBuffers,
    TierCeiling, Toom8,
};

const PART_COUNTS: [NonZeroUsize; 3] = [
    NonZeroUsize::new(Toom8::BALANCED_PARTS).unwrap(),
    NonZeroUsize::new(Toom8::HALF_LARGE_PARTS).unwrap(),
    NonZeroUsize::new(Toom8::HALF_SMALL_PARTS).unwrap(),
];

/// Destination or scratch storage for the 7 non-endpoint Toom-8 points.
pub struct BasePoints<'buffer> {
    pub one: &'buffer mut [Limb],
    pub two: &'buffer mut [Limb],
    pub four: &'buffer mut [Limb],
    pub eight: &'buffer mut [Limb],
    pub half: &'buffer mut [Limb],
    pub quarter: &'buffer mut [Limb],
    pub eighth: &'buffer mut [Limb],
}

pub struct DestinationPoints<'buffer, Output = Limb> {
    pub zero: &'buffer [Output],
    pub half: &'buffer mut [Output],
    pub one: &'buffer mut [Output],
    pub four: &'buffer mut [Output],
    pub infinity: &'buffer [Output],
}

impl Toom8 {
    /// Computes the positive split of an already admitted Toom-8/8.5 shape.
    /// Sizing queries reject empty/ineligible widths before reaching this geometry.
    pub fn multiplication_split_len(shape: MulShape, len_a: usize, len_b: usize) -> NonZeroUsize {
        let smaller = min(len_a, len_b);
        let larger = max(len_a, len_b);
        // SAFETY: the driver and sizing facade enter only after Toom-8 shape
        // admission, or eight-part square admission. Both prove larger>=8.
        let larger_width = unsafe { NonZeroUsize::new_unchecked(larger) };
        match shape {
            MulShape::Balanced => larger_width.div_ceil(PART_COUNTS[0]),
            MulShape::Half => {
                // SAFETY: half admission additionally proves the smaller
                // operand contains eight nonempty parts, hence smaller>=8.
                let smaller_width = unsafe { NonZeroUsize::new_unchecked(smaller) };
                max(
                    larger_width.div_ceil(PART_COUNTS[1]),
                    smaller_width.div_ceil(PART_COUNTS[2]),
                )
            }
        }
    }

    pub fn local_mul_scratch_len(
        shape: MulShape,
        split_width: NonZeroUsize,
        len_a: usize,
        len_b: usize,
    ) -> usize {
        let split_len = split_width.get();
        let product_len = len_a
            .checked_add(len_b)
            .expect("Toom-8 product exceeds usize");
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let mut inner = ChildDemand::recursive_mul_scratch(split_len, eval_len);
        let mut infinity_len = 0;
        if matches!(shape, MulShape::Half) {
            let larger = max(len_a, len_b);
            let smaller = min(len_a, len_b);
            // SAFETY: dispatch admitted larger>8m and smaller>7m, even for
            // virtual widths. Those offsets fit their respective widths, and
            // the suffixes' sum is below the checked complete product width.
            let (high_large, high_small) = unsafe {
                (
                    larger.unchecked_sub(split_len.unchecked_mul(8)),
                    smaller.unchecked_sub(split_len.unchecked_mul(7)),
                )
            };
            // SAFETY: the two suffix widths sum to product_len-15m<=product_len.
            infinity_len = unsafe { high_large.unchecked_add(high_small) };
            let plan = Multiplication::select_plan(high_large, high_small, TierCeiling::Toom6);
            inner = max(
                inner,
                Multiplication::scratch_len(plan, high_large, high_small),
            );
        }
        // SAFETY: m<=ceil(usize::MAX/8)=2^(w-3), and interpolation guards g<=4;
        // hence value_len+m=3m+g<usize::MAX on w in {16,32,64}.
        let packed_len = unsafe { value_len.unchecked_add(split_len) };
        let points_are_placed =
            Self::destination_points_fit(product_len, split_len, packed_len, infinity_len);
        // p*(3m+g)+4*(m+e) = (3p+4)m+pg+4e, with p=5/8, g<=4, e<=2.
        // SAFETY: these two fixed layouts give factor=19/28 and guards<=40,
        // independently of operand width. Only the final allocation total can overflow.
        let (factor, guards) = unsafe {
            let points: usize = if points_are_placed { 5 } else { 8 };
            (
                if points_are_placed { 19 } else { 28 },
                points
                    .unchecked_mul(INTERPOLATION_GUARD_LIMBS)
                    .unchecked_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS).unchecked_mul(4)),
            )
        };
        split_len
            .checked_mul(factor)
            .and_then(|width| width.checked_add(guards))
            .and_then(|width| width.checked_add(inner))
            .expect("Toom-8 workspace exceeds usize")
    }

    pub fn local_sqr_scratch_len(len: usize) -> usize {
        let product_len = len.checked_mul(2).expect("Toom-8 square exceeds usize");
        let split_width = Self::multiplication_split_len(MulShape::Balanced, len, len);
        let split_len = split_width.get();
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let inner = ChildDemand::recursive_sqr_scratch(split_len, eval_len);
        // SAFETY: m=ceil(len/8)<=2^(w-3), and g<=4, so 3m+g fits usize.
        let packed_len = unsafe { value_len.unchecked_add(split_len) };
        let points_are_placed = Self::destination_points_fit(product_len, split_len, packed_len, 0);
        // p*(3m+g)+2*(m+e) = (3p+2)m+pg+2e, with p=5/8.
        // SAFETY: the fixed layouts give factor=17/26 and guards<=36 at 16 bits.
        let (factor, guards) = unsafe {
            let points: usize = if points_are_placed { 5 } else { 8 };
            (
                if points_are_placed { 17 } else { 26 },
                points
                    .unchecked_mul(INTERPOLATION_GUARD_LIMBS)
                    .unchecked_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS).unchecked_mul(2)),
            )
        };
        split_len
            .checked_mul(factor)
            .and_then(|width| width.checked_add(guards))
            .and_then(|width| width.checked_add(inner))
            .expect("Toom-8 square workspace exceeds usize")
    }

    #[inline]
    pub fn split_mul_scratch(
        scratch: &mut [Limb],
        packed_len: usize,
        eval_len: usize,
        points_are_placed: bool,
        fast_paired_add_sub: bool,
        add_mul_kernel: AddMulKernel,
    ) -> (BasePoints<'_>, &mut [Limb], MulEvaluationBuffers<'_>) {
        let scratch_point_len = if points_are_placed { 0 } else { packed_len };
        // SAFETY: local_mul_scratch_len checked p*packed_len+4*eval_len+inner,
        // with p=5 when placement passed and p=8 otherwise. The driver reuses
        // that placement decision; every consecutive split stays in initialized
        // scratch and creates disjoint live point, evaluation, and child spans.
        unsafe {
            let (one, after_one) = scratch.split_at_mut_unchecked(scratch_point_len);
            let (two, after_two) = after_one.split_at_mut_unchecked(packed_len);
            let (four, after_four) = after_two.split_at_mut_unchecked(scratch_point_len);
            let (eight, after_eight) = after_four.split_at_mut_unchecked(packed_len);
            let (half, after_half) = after_eight.split_at_mut_unchecked(scratch_point_len);
            let (quarter, after_quarter) = after_half.split_at_mut_unchecked(packed_len);
            let (eighth, after_eighth) = after_quarter.split_at_mut_unchecked(packed_len);
            let (temporary, after_temporary) = after_eighth.split_at_mut_unchecked(packed_len);
            let (eval_a, after_eval_a) = after_temporary.split_at_mut_unchecked(eval_len);
            let (eval_b, after_eval_b) = after_eval_a.split_at_mut_unchecked(eval_len);
            let (odd_a, after_odd_a) = after_eval_b.split_at_mut_unchecked(eval_len);
            let (odd_b, scratch_eval) = after_odd_a.split_at_mut_unchecked(eval_len);
            (
                BasePoints {
                    one,
                    two,
                    four,
                    eight,
                    half,
                    quarter,
                    eighth,
                },
                temporary,
                MulEvaluationBuffers {
                    eval_a,
                    eval_b,
                    odd_a,
                    odd_b,
                    scratch: scratch_eval,
                    fast_paired_add_sub,
                    add_mul_kernel,
                },
            )
        }
    }

    #[inline]
    pub fn split_sqr_scratch(
        scratch: &mut [Limb],
        packed_len: usize,
        eval_len: usize,
        points_are_placed: bool,
        fast_paired_add_sub: bool,
        add_mul_kernel: AddMulKernel,
    ) -> (BasePoints<'_>, &mut [Limb], SqrEvaluationBuffers<'_>) {
        let scratch_point_len = if points_are_placed { 0 } else { packed_len };
        // SAFETY: local_sqr_scratch_len checked p*packed_len+2*eval_len+inner,
        // with p=5/8 for the admitted placement/fallback. Each consecutive split
        // produces initialized disjoint storage, consuming exactly that local sum.
        unsafe {
            let (one, after_one) = scratch.split_at_mut_unchecked(scratch_point_len);
            let (two, after_two) = after_one.split_at_mut_unchecked(packed_len);
            let (four, after_four) = after_two.split_at_mut_unchecked(scratch_point_len);
            let (eight, after_eight) = after_four.split_at_mut_unchecked(packed_len);
            let (half, after_half) = after_eight.split_at_mut_unchecked(scratch_point_len);
            let (quarter, after_quarter) = after_half.split_at_mut_unchecked(packed_len);
            let (eighth, after_eighth) = after_quarter.split_at_mut_unchecked(packed_len);
            let (temporary, after_temporary) = after_eighth.split_at_mut_unchecked(packed_len);
            let (eval, after_eval) = after_temporary.split_at_mut_unchecked(eval_len);
            let (odd, scratch_eval) = after_eval.split_at_mut_unchecked(eval_len);
            (
                BasePoints {
                    one,
                    two,
                    four,
                    eight,
                    half,
                    quarter,
                    eighth,
                },
                temporary,
                SqrEvaluationBuffers {
                    eval,
                    odd,
                    scratch: scratch_eval,
                    fast_paired_add_sub,
                    add_mul_kernel,
                },
            )
        }
    }

    pub const fn evaluation_len(split_len: usize) -> usize {
        debug_assert!(
            split_len <= usize::MAX.div_ceil(8),
            "evaluation split exceeds the admitted virtual width"
        );
        // SAFETY: all callers use m<=ceil(usize::MAX/8)=2^(w-3), and the
        // evaluation guard is at most two limbs. Their sum fits w>=16.
        unsafe { split_len.unchecked_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS)) }
    }

    pub const fn interpolation_value_len(split_len: usize) -> usize {
        // Point products are <2^49*B^(2m). Every signed packed matrix row is
        // <1_256_584_717_458*B^(3m), requiring 42 guard bits above its packed
        // radix unit. Two evaluation guards retain the complete recursive
        // output and provide >=64 guard bits on every supported pointer width.
        debug_assert!(
            split_len <= usize::MAX.div_ceil(8),
            "interpolation split exceeds the admitted virtual width"
        );
        // SAFETY: m<=2^(w-3) and guards<=4, so 2m+guards<2^w for w>=16.
        unsafe {
            split_len
                .unchecked_mul(2)
                .unchecked_add(INTERPOLATION_GUARD_LIMBS)
        }
    }

    /// Partition a placement already admitted by `destination_points_fit`.
    pub const fn split_destination_points<Output>(
        dst: &mut [Output],
        split_len: usize,
        packed_len: usize,
        infinity_len: usize,
    ) -> DestinationPoints<'_, Output> {
        debug_assert!(
            Self::destination_points_fit(dst.len(), split_len, packed_len, infinity_len),
            "destination placement was not admitted before partitioning"
        );
        // SAFETY: both drivers validated 11m+packed_len <= dst.len() and
        // packed_len <= 4m before partitioning scratch. Therefore 2m, 3m,
        // 4m-packed_len, and every point interval fit without overflow.
        // Sequential splits produce disjoint initialized endpoint/point slices.
        let (zero, half, one, four, after_four) = unsafe {
            let zero_len = split_len.unchecked_mul(2);
            let half_offset = split_len.unchecked_mul(3);
            let gap = split_len.unchecked_mul(4).unchecked_sub(packed_len);
            let (before_half, half_and_after) = dst.split_at_mut_unchecked(half_offset);
            let (half, after_half) = half_and_after.split_at_mut_unchecked(packed_len);
            let (_, one_and_after) = after_half.split_at_mut_unchecked(gap);
            let (one, after_one) = one_and_after.split_at_mut_unchecked(packed_len);
            let (_, four_and_after) = after_one.split_at_mut_unchecked(gap);
            let (four, after_four) = four_and_after.split_at_mut_unchecked(packed_len);
            let (zero, _) = before_half.split_at_unchecked(zero_len);
            (zero, half, one, four, after_four)
        };
        let infinity = if infinity_len == 0 {
            &[]
        } else {
            // SAFETY: nonzero infinity admission additionally proved
            // 15m+infinity_len <= dst.len(); the same 4m-packed_len gap
            // separates the four-point buffer from that initialized endpoint.
            unsafe {
                let gap = split_len.unchecked_mul(4).unchecked_sub(packed_len);
                let (_, infinity_and_after) = after_four.split_at_unchecked(gap);
                infinity_and_after.split_at_unchecked(infinity_len).0
            }
        };
        DestinationPoints {
            zero,
            half,
            one,
            four,
            infinity,
        }
    }

    pub const fn destination_points_fit(
        product_len: usize,
        split_len: usize,
        packed_len: usize,
        infinity_len: usize,
    ) -> bool {
        // Consecutive point offsets differ by 4m. The zero product ends at
        // 2m <= 3m; all three point-disjointness checks reduce to packed_len <= 4m.
        let Some(spacing) = split_len.checked_mul(4) else {
            return false;
        };
        if packed_len > spacing {
            return false;
        }
        let Some(four_offset) = split_len.checked_mul(11) else {
            return false;
        };
        let Some(placed_end) = four_offset.checked_add(packed_len) else {
            return false;
        };
        if placed_end > product_len {
            return false;
        }
        if infinity_len == 0 {
            return true;
        }
        // packed_len <= 4m already gives placed_end <= 15m.
        let Some(infinity_offset) = split_len.checked_mul(15) else {
            return false;
        };
        let Some(after_infinity) = infinity_offset.checked_add(infinity_len) else {
            return false;
        };
        after_infinity <= product_len
    }

    pub fn clear_destination_gaps(
        dst: &mut [impl LimbOutput],
        split_len: usize,
        packed_len: usize,
        zero_product_len: usize,
        infinity_len: usize,
        points_are_placed: bool,
    ) {
        let upper_gap_end = if infinity_len == 0 {
            dst.len()
        } else {
            // SAFETY: nonempty infinity proves 15m<|a|+|b|<=dst.len().
            unsafe { split_len.unchecked_mul(15) }
        };
        if !points_are_placed {
            clear_range(dst, zero_product_len, upper_gap_end);
            if infinity_len != 0 {
                // SAFETY: nonempty infinity occupies [15m,|a|+|b|), within dst.
                let after_infinity = unsafe { upper_gap_end.unchecked_add(infinity_len) };
                clear_range(dst, after_infinity, dst.len());
            }
            return;
        }

        // The point buffers at 3m, 7m, and 11m overwrite their complete packed
        // ranges. Initialize only the disjoint gaps that coefficient additions can
        // observe, preserving the endpoint products at shifts zero and fifteen.
        // SAFETY: placement validated packed_len<=4m and 11m+packed_len<=dst.len().
        // The increasing 3m/7m/11m intervals therefore fit without overflow.
        let (half_offset, one_offset, four_offset, after_half, after_one, after_four) = unsafe {
            let half = split_len.unchecked_mul(3);
            let one = split_len.unchecked_mul(7);
            let four = split_len.unchecked_mul(11);
            (
                half,
                one,
                four,
                half.unchecked_add(packed_len),
                one.unchecked_add(packed_len),
                four.unchecked_add(packed_len),
            )
        };
        clear_range(dst, zero_product_len, half_offset);
        clear_range(dst, after_half, one_offset);
        clear_range(dst, after_one, four_offset);
        clear_range(dst, after_four, upper_gap_end);
        if infinity_len != 0 {
            // SAFETY: the admitted infinity endpoint ends within the full product.
            let after_infinity = unsafe { upper_gap_end.unchecked_add(infinity_len) };
            clear_range(dst, after_infinity, dst.len());
        }
    }

    pub fn multiply_endpoints(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        shape: MulShape,
        split_len: usize,
    ) -> (usize, usize) {
        // SAFETY: both balanced and half admission prove |a|,|b|>7m, so the
        // low m-limb parts exist and their 2m-limb product fits the full output.
        let (low_a, low_b, zero_product_len, zero_product) = unsafe {
            let zero_len = split_len.unchecked_mul(2);
            (
                a.split_at_unchecked(split_len).0,
                b.split_at_unchecked(split_len).0,
                zero_len,
                dst.split_at_mut_unchecked(zero_len).0,
            )
        };
        Recursive::recursive_mul(zero_product, low_a, low_b, scratch, TierCeiling::Toom6);

        if matches!(shape, MulShape::Balanced) {
            return (zero_product_len, 0);
        }

        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        // SAFETY: half admission gives |larger|>8m and |smaller|>7m.
        // Their nonempty suffix product occupies [15m,|a|+|b|), inside dst.
        let (high_large, high_small, product_len, product) = unsafe {
            let high_large = larger.split_at_unchecked(split_len.unchecked_mul(8)).1;
            let high_small = smaller.split_at_unchecked(split_len.unchecked_mul(7)).1;
            let product_len = high_large.len().unchecked_add(high_small.len());
            let product = dst
                .split_at_mut_unchecked(split_len.unchecked_mul(15))
                .1
                .split_at_mut_unchecked(product_len)
                .0;
            (high_large, high_small, product_len, product)
        };
        Recursive::recursive_mul(product, high_large, high_small, scratch, TierCeiling::Toom6);
        (zero_product_len, product_len)
    }
}

fn clear_range(dst: &mut [impl LimbOutput], start: usize, end: usize) {
    debug_assert!(
        start <= end && end <= dst.len(),
        "clear range exceeds product"
    );
    // SAFETY: callers partition only endpoint gaps or point gaps whose increasing
    // start/end bounds were established by shape admission and placement validation.
    unsafe { dst.get_unchecked_mut(start..end) }.fill(LimbOutput::from_limb(0));
}
