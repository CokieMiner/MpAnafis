//! The Toom-Cook 6 and 6.5 drivers: split, evaluate, interpolate, reconstruct.

#![expect(
    unsafe_code,
    reason = "Validated split geometry bounds disjoint scratch partitions, initialized endpoints, and packed destination points"
)]

use core::cmp::max;

use super::{
    ArchKernels, Limb, LimbOutput, MulEvaluationBuffers, MulShape, Multiplication, Recursive,
    SharedEval, SqrEvaluationBuffers, TierCeiling, Values, Widths,
};

/// Namespace for the six-way and six-and-a-half-way Toom-Cook tiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom6;

pub struct ProductPair<'buffer, Output = Limb> {
    pub positive: &'buffer mut [Output],
    pub negative: &'buffer mut [Output],
}

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
    /// Multiply two balanced limb slices with a six-way Toom-Cook split.
    pub fn mul<Output: LimbOutput>(
        dst: &mut [Output],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        if a.len() < 6 || b.len() < 6 {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
            return;
        }
        let Some(shape) = Widths::new(a.len(), b.len()).toom6_shape() else {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom4);
            return;
        };
        if matches!(shape, MulShape::Half) {
            Self::mul_half(dst, a, b, scratch);
            return;
        }

        let split_len = max(a.len(), b.len()).div_ceil(6);
        // SAFETY: m=ceil(maximum actual limb-slice length/6). The isize::MAX
        // byte bound and >=2 bytes per limb leave room for 2m+2 and m+1.
        let (eval_len, value_len) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_mul(2).unchecked_add(2),
            )
        };
        debug_assert!(
            scratch.len() >= Multiplication::toom6_mul_scratch_len(a.len(), b.len()),
            "Toom-6 multiplication scratch is undersized"
        );
        let place_alternating = Self::destination_points_fit(dst.len(), split_len);
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
        } = Self::split_scratch::<false>(scratch, value_len, eval_len, place_alternating);
        let parts_a = Self::split_six(a, split_len);
        let parts_b = Self::split_six(b, split_len);

        // Preserve c0 directly in the output. The recursive product initializes
        // its complete endpoint, while the two placed interpolation buffers also
        // overwrite their complete ranges, so clear only the remaining canvas.
        // SAFETY: both constant blocks are full m-limb prefixes in admitted
        // balanced shapes, so their exact product width is 2m<=dst.len().
        let zero_product_len = unsafe { split_len.unchecked_mul(2) };
        Self::clear_destination(dst, zero_product_len, split_len, place_alternating);
        // SAFETY: the zero endpoint is a prefix of the full product destination.
        let (zero_product, _) = unsafe { dst.split_at_mut_unchecked(zero_product_len) };
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
        // SAFETY: these scratch views already contain initialized limbs. Every
        // evaluation/coupling operation only writes initialized limb values;
        // no path deinitializes storage before the original scratch borrow resumes.
        let output_values = unsafe {
            Values {
                one: Output::from_initialized_mut(one),
                two: Output::from_initialized_mut(two),
                four: Output::from_initialized_mut(four),
                half: Output::from_initialized_mut(half),
                quarter: Output::from_initialized_mut(quarter),
            }
        };
        Self::evaluate_and_reconstruct(
            dst,
            split_len,
            place_alternating,
            output_values,
            |values, zero| {
                Self::evaluate_mul_points(
                    values,
                    temporary,
                    &mut evaluations,
                    parts_a,
                    parts_b,
                    zero,
                    split_len,
                );
            },
        );
    }

    /// Square a limb slice with a six-way Toom-Cook split.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 6 {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom4);
            return;
        }

        let split_len = a.len().div_ceil(6);
        // SAFETY: m=ceil(|a|/6) for a real limb slice; its byte bound leaves
        // room for m+1 and 2m+2 on 16-, 32-, and 64-bit targets.
        let (eval_len, value_len) = unsafe {
            (
                split_len.unchecked_add(1),
                split_len.unchecked_mul(2).unchecked_add(2),
            )
        };
        let place_alternating = Self::destination_points_fit(dst.len(), split_len);
        debug_assert!(
            dst.len() >= a.len().saturating_mul(2),
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
        } = Self::split_scratch::<true>(scratch, value_len, eval_len, place_alternating);
        let parts = Self::split_six(a, split_len);

        // The endpoint square and placed tables overwrite their complete ranges.
        // Only gaps in the reconstruction canvas require an initial zero fill.
        // SAFETY: the constant block is a full m-limb prefix, with 2m<=dst.len().
        let zero_product_len = unsafe { split_len.unchecked_mul(2) };
        Self::clear_destination(dst, zero_product_len, split_len, place_alternating);
        // SAFETY: the exact zero square fits the validated destination prefix.
        let (zero_product, _) = unsafe { dst.split_at_mut_unchecked(zero_product_len) };
        Recursive::recursive_sqr(zero_product, parts.constant, inner, TierCeiling::Toom4);
        let mut evaluations = SqrEvaluationBuffers {
            eval: eval_a,
            odd: odd_a,
            scratch: inner,
            add_mul_kernel: ArchKernels::selected_add_mul_limbs_unchecked(),
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
        };
        Self::evaluate_and_reconstruct(
            dst,
            split_len,
            place_alternating,
            Values {
                one,
                two,
                four,
                half,
                quarter,
            },
            |values, zero| {
                Self::evaluate_sqr_points(
                    values,
                    temporary,
                    &mut evaluations,
                    parts,
                    zero,
                    split_len,
                );
            },
        );
    }

    /// Partitions packed point tables, evaluations, and recursive scratch.
    pub const fn split_scratch<const SQUARE: bool>(
        scratch: &mut [Limb],
        value_len: usize,
        eval_len: usize,
        points_are_placed: bool,
    ) -> ScratchLayout<'_> {
        // SAFETY: checked local sizing reserves p*(3m+2)+(2m+2)+k*(m+1)
        // limbs plus children, with p=3/5 and k=2/4. eval_len=m+1>=2,
        // value_len=2m+2, so the derived packed width fits that actual scratch.
        let packed_len = unsafe { value_len.unchecked_add(eval_len.unchecked_sub(1)) };
        let scratch_point_len = if points_are_placed { 0 } else { packed_len };
        let second_eval_len = if SQUARE { 0 } else { eval_len };
        // SAFETY: the checked total above accounts for every consecutive split.
        // Zero-length placed-point/second-operand windows consume no storage;
        // all other initialized spans remain disjoint with child scratch last.
        let (one, two, four, half, quarter, temporary, eval_a, eval_b, odd_a, odd_b, inner) = unsafe {
            let (one, after_one) = scratch.split_at_mut_unchecked(packed_len);
            let (two, after_two) = after_one.split_at_mut_unchecked(scratch_point_len);
            let (four, after_four) = after_two.split_at_mut_unchecked(packed_len);
            let (half, after_half) = after_four.split_at_mut_unchecked(scratch_point_len);
            let (quarter, after_quarter) = after_half.split_at_mut_unchecked(packed_len);
            let (temporary, after_temporary) = after_quarter.split_at_mut_unchecked(value_len);
            let (eval_a, after_eval_a) = after_temporary.split_at_mut_unchecked(eval_len);
            let (eval_b, after_eval_b) = after_eval_a.split_at_mut_unchecked(second_eval_len);
            let (odd_a, after_odd_a) = after_eval_b.split_at_mut_unchecked(eval_len);
            let (odd_b, inner) = after_odd_a.split_at_mut_unchecked(second_eval_len);
            (
                one, two, four, half, quarter, temporary, eval_a, eval_b, odd_a, odd_b, inner,
            )
        };

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

    /// Reserves only live point buffers and one operand's evaluations per square.
    pub fn local_scratch_len<const SQUARE: bool>(
        split_len: usize,
        inner_space: usize,
        points_are_placed: bool,
    ) -> usize {
        // p packed (3m+2)-limb tables, one (2m+2)-limb temporary,
        // and k (m+1)-limb evaluations give (3p+2+k)m+(2p+2+k),
        // with p in {3,5} and k in {2,4}.
        // Evaluation products are <5461^2*B^(2m)<2^25*B^(2m); the largest
        // signed matrix row is <978670*B^(3m), requiring 21 guard bits.
        // Two guard limbs provide >=32 bits and retain the full recursive output.
        let (factor, guards) = if SQUARE {
            if points_are_placed {
                (13, 10)
            } else {
                (19, 14)
            }
        } else if points_are_placed {
            (15, 12)
        } else {
            (21, 16)
        };
        split_len
            .checked_mul(factor)
            .and_then(|width| width.checked_add(guards))
            .and_then(|width| width.checked_add(inner_space))
            .expect("Toom-6 workspace exceeds usize")
    }

    /// Packed windows at offsets `3m` and `7m` are disjoint when `m >= 2`.
    /// The second ends at `10m + 2`, which must fit the complete destination.
    pub const fn destination_points_fit(product_len: usize, split_len: usize) -> bool {
        if split_len < 2 {
            return false;
        }
        let Some(offset) = split_len.checked_mul(10) else {
            return false;
        };
        let Some(end) = offset.checked_add(2) else {
            return false;
        };
        end <= product_len
    }
    fn clear_destination(
        dst: &mut [impl LimbOutput],
        zero_product_len: usize,
        split_len: usize,
        place_alternating: bool,
    ) {
        if !place_alternating {
            // SAFETY: zero_product_len=2m is the prefix endpoint's exact width.
            let (_, reconstruction_canvas) =
                unsafe { dst.split_at_mut_unchecked(zero_product_len) };
            reconstruction_canvas.fill(LimbOutput::from_limb(0));
            return;
        }

        // SAFETY: placement proves m>=2 and 10m+2<=dst.len(). Thus [3m,6m+2)
        // and [7m,10m+2) fit and are disjoint, separated by m-2 limbs. The zero
        // endpoint ends at 2m; all cleared spans are the disjoint reconstruction gaps.
        let (clear_before_two, gap, after_half) = unsafe {
            let packed_len = split_len.unchecked_mul(3).unchecked_add(2);
            let (before_two, two_and_after) =
                dst.split_at_mut_unchecked(split_len.unchecked_mul(3));
            let clear_before_two = before_two.split_at_mut_unchecked(zero_product_len).1;
            let after_two = two_and_after.split_at_mut_unchecked(packed_len).1;
            let (gap, half_and_after) =
                after_two.split_at_mut_unchecked(split_len.unchecked_sub(2));
            (
                clear_before_two,
                gap,
                half_and_after.split_at_mut_unchecked(packed_len).1,
            )
        };
        clear_before_two.fill(LimbOutput::from_limb(0));
        gap.fill(LimbOutput::from_limb(0));
        after_half.fill(LimbOutput::from_limb(0));
    }

    /// Places two packed tables directly at their final radix positions.
    ///
    /// Multiplication and squaring have identical interpolation geometry. The
    /// evaluator fills every table before interpolation, while the constant
    /// endpoint remains disjoint in the low destination prefix.
    fn evaluate_and_reconstruct<Output: LimbOutput>(
        dst: &mut [Output],
        split_len: usize,
        place_alternating: bool,
        mut values: Values<'_, Output>,
        evaluate: impl FnOnce(&mut Values<'_, Output>, &[Limb]),
    ) {
        // SAFETY: each admitted operand has a full m-limb constant block, so
        // zero_len=2m is bounded by the validated complete product destination.
        let zero_len = unsafe { split_len.unchecked_mul(2) };
        if !place_alternating {
            // SAFETY: the initialized zero endpoint occupies the 2m-limb prefix.
            let zero = unsafe { Output::assume_init(dst.split_at_unchecked(zero_len).0) };
            evaluate(&mut values, zero);
            // SAFETY: endpoint recursion and clear_destination initialized the whole
            // destination; evaluation and coupling initialized every packed table.
            unsafe {
                Self::interpolate_and_reconstruct(
                    Output::assume_init_mut(dst),
                    split_len,
                    values.assume_init(),
                );
            }
            return;
        }

        let Values {
            one, four, quarter, ..
        } = values;
        {
            // SAFETY: placement proves m>=2 and 10m+2<=dst.len(). The
            // [3m,6m+2) and [7m,10m+2) windows are disjoint and in bounds,
            // with gap m-2; the initialized zero endpoint ends at 2m<3m.
            let (zero, placed_two, placed_half) = unsafe {
                let packed_len = split_len.unchecked_mul(3).unchecked_add(2);
                let (before_two, two_and_after) =
                    dst.split_at_mut_unchecked(split_len.unchecked_mul(3));
                let (placed_two, after_two) = two_and_after.split_at_mut_unchecked(packed_len);
                let half_and_after = after_two
                    .split_at_mut_unchecked(split_len.unchecked_sub(2))
                    .1;
                (
                    Output::assume_init(before_two.split_at_unchecked(zero_len).0),
                    placed_two,
                    half_and_after.split_at_mut_unchecked(packed_len).0,
                )
            };
            let mut placed = Values {
                one: &mut *one,
                two: placed_two,
                four: &mut *four,
                half: placed_half,
                quarter: &mut *quarter,
            };
            evaluate(&mut placed, zero);
            // SAFETY: every conjugate product writes its full high suffix, and
            // coupling writes the disjoint low prefix of all five packed tables.
            Self::interpolate_values(unsafe { placed.assume_init() });
        }
        // SAFETY: placed intervals end at 10m+2<=dst.len(), bounding 5m and 9m.
        let (fifth_offset, ninth_offset) =
            unsafe { (split_len.unchecked_mul(5), split_len.unchecked_mul(9)) };
        // SAFETY: endpoint recursion, disjoint gap fills, and both placed coupled
        // tables cover the complete destination. The evaluator initialized the
        // three scratch tables as well; all placed borrows ended before this view.
        let (initialized, one_value, four_value, quarter_value) = unsafe {
            (
                Output::assume_init_mut(dst),
                Output::assume_init(one),
                Output::assume_init(four),
                Output::assume_init(quarter),
            )
        };
        SharedEval::add_coefficient_in_place(initialized, four_value, split_len);
        SharedEval::add_coefficient_in_place(initialized, one_value, fifth_offset);
        SharedEval::add_coefficient_in_place(initialized, quarter_value, ninth_offset);
    }
}
