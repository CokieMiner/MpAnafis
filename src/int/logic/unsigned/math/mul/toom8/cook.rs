//! The Toom-Cook 8 and 8.5 drivers: split, couple, interpolate, reconstruct.

#![expect(
    unsafe_code,
    reason = "Eight-/nine-part admission and checked workspace totals bound initialized endpoints, disjoint point buffers, and guard expansion"
)]

use super::{
    AddMulKernel, ArchKernels, BasePoints, CouplingContext, LIMB_BITS, Limb, LimbOutput, MulShape,
    Multiplication, Recursive, SharedEval, TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, TierCeiling,
    Values, Widths,
};

// The largest finite evaluation is at |x|=8 with nine coefficients below B^m.
// Its multiplier is 1+8+...+8^8=19_173_961, whose exact bit width is 25.
#[expect(
    clippy::as_conversions,
    reason = "the fixed nine-term bound has 25 bits, which fits usize on every supported target"
)]
pub const EVALUATION_GUARD_BITS: usize = {
    let mut multiplier = 1_u32;
    let mut terms = 1_usize;
    while terms < Toom8::HALF_LARGE_PARTS {
        multiplier = multiplier
            .checked_mul(8)
            .expect("fixed evaluation bound fits u32")
            .checked_add(1)
            .expect("fixed evaluation bound fits u32");
        terms = terms.checked_add(1).expect("fixed coefficient count fits");
    }
    (u32::BITS - multiplier.leading_zeros()) as usize
};
// Point products require the complete 2*(m+e)-limb recursive output, where
// e=ceil(25/LIMB_BITS). Each product is <2^49*B^(2m); the largest signed
// interpolation row is <1_256_584_717_458*B^(3m), requiring 42 guard bits.
// Two evaluation guards provide >=64 bits on every supported target.
// SAFETY: supported limbs have 16/32/64 bits, giving e in {1,2} and 2e<=4.
pub const INTERPOLATION_GUARD_LIMBS: usize =
    unsafe { EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS).unchecked_mul(2) };

/// Namespace for the eight-way and eight-and-a-half-way Toom-Cook tiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom8;

pub struct ProductPair<'buffer, Output = Limb> {
    pub positive: &'buffer mut [Output],
    pub negative: &'buffer mut [Output],
}

impl Toom8 {
    /// Polynomial coefficient count of the balanced kernel.
    pub const BALANCED_PARTS: usize = 8;
    /// Longer polynomial coefficient count of the adjacent kernel.
    pub const HALF_LARGE_PARTS: usize = Self::BALANCED_PARTS + 1;
    /// Shorter polynomial coefficient count of the adjacent kernel.
    pub const HALF_SMALL_PARTS: usize = Self::BALANCED_PARTS;

    /// Multiply with a balanced Toom-8 or adjacent unbalanced Toom-8.5 split.
    #[expect(
        clippy::too_many_lines,
        reason = "The driver keeps one linear split-evaluate-interpolate sequence whose buffers and lifetimes are coupled"
    )]
    pub fn mul<Output: LimbOutput>(
        dst: &mut [Output],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
    ) {
        let Some(shape) = Widths::new(a.len(), b.len()).toom8_shape() else {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom6);
            return;
        };
        debug_assert!(
            dst.len() >= a.len().saturating_add(b.len()),
            "Toom-8 multiplication output is shorter than the full product"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom8_mul_scratch_len(a.len(), b.len()),
            "Toom-8 multiplication scratch buffer is undersized"
        );

        let split_width = Self::multiplication_split_len(shape, a.len(), b.len());
        let split_len = split_width.get();
        let degree = match shape {
            MulShape::Balanced => 14,
            MulShape::Half => 15,
        };
        // SAFETY: m is ceil(maximum actual operand length/8) or the admitted
        // nine-/eight-part width. The isize::MAX byte bound leaves room for
        // m+ceil(25/LIMB_BITS) and the packed width 3m+2*ceil(25/LIMB_BITS).
        let (eval_len, packed_len) = unsafe {
            (
                split_len.unchecked_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS)),
                split_len
                    .unchecked_mul(3)
                    .unchecked_add(INTERPOLATION_GUARD_LIMBS),
            )
        };
        let expected_infinity_len = if matches!(shape, MulShape::Half) {
            // SAFETY: half admission gives |larger|>8m and |smaller|>7m;
            // their full width fits dst and exceeds the 15m infinity offset.
            unsafe {
                a.len()
                    .unchecked_add(b.len())
                    .unchecked_sub(split_len.unchecked_mul(15))
            }
        } else {
            0
        };
        let place_points =
            Self::destination_points_fit(dst.len(), split_len, packed_len, expected_infinity_len);
        let fast_paired_add_sub = ArchKernels::fast_add_sub_limbs_available();
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();
        let (points, temporary, mut evaluations) = Self::split_mul_scratch(
            scratch,
            packed_len,
            eval_len,
            place_points,
            fast_paired_add_sub,
            add_mul_kernel,
        );
        let (zero_product_len, infinity_len) =
            Self::multiply_endpoints(dst, a, b, evaluations.scratch, shape, split_len);
        debug_assert_eq!(
            infinity_len, expected_infinity_len,
            "Toom-8.5 endpoint geometry disagrees with scratch sizing"
        );
        Self::clear_destination_gaps(
            dst,
            split_len,
            packed_len,
            zero_product_len,
            infinity_len,
            place_points,
        );
        let BasePoints {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
        } = points;
        if place_points {
            let placed = Self::split_destination_points(dst, split_len, packed_len, infinity_len);
            // SAFETY: scratch tables already contain initialized limbs. Product
            // writers and coupling only store initialized values; every path,
            // including unwinding, preserves the original scratch validity.
            let mut values = unsafe {
                Values {
                    one: placed.one,
                    two: Output::from_initialized_mut(two),
                    four: placed.four,
                    eight: Output::from_initialized_mut(eight),
                    half: placed.half,
                    quarter: Output::from_initialized_mut(quarter),
                    eighth: Output::from_initialized_mut(eighth),
                }
            };
            // SAFETY: multiply_endpoints initialized both complete endpoint
            // spans before destination partitioning; empty infinity is readable.
            let (zero_value, infinity_value) = unsafe {
                (
                    Output::assume_init(placed.zero),
                    Output::assume_init(placed.infinity),
                )
            };
            let context = CouplingContext {
                zero: zero_value,
                infinity: infinity_value,
                split_len: split_width,
                degree,
            };
            Self::evaluate_and_couple_mul(&mut values, temporary, &mut evaluations, a, b, &context);
            // SAFETY: every product writer initialized the complete high suffix;
            // coupling initialized each low prefix, including all placed tables.
            Self::interpolate_values(unsafe { values.assume_init() }, temporary, add_mul_kernel);
            // SAFETY: endpoint writers, all disjoint gap fills and the three
            // placed coupled tables now cover the complete destination.
            let initialized = unsafe { Output::assume_init_mut(dst) };
            Self::reconstruct_alternating(
                initialized,
                split_len,
                [eighth, quarter, two, eight],
                fast_paired_add_sub,
            );
        } else {
            // SAFETY: without placed points, the endpoint writers and gap fills
            // already cover every destination limb; all point tables use scratch.
            let initialized = unsafe { Output::assume_init_mut(dst) };
            // SAFETY: multiply_endpoints initialized this exact 2m-limb prefix.
            let zero = unsafe { initialized.split_at_unchecked(zero_product_len).0 };
            let infinity = if infinity_len == 0 {
                &[]
            } else {
                // SAFETY: the nonempty half endpoint occupies [15m,|a|+|b|),
                // contained in the validated full output and disjoint from zero.
                unsafe {
                    initialized
                        .split_at_unchecked(split_len.unchecked_mul(15))
                        .1
                        .split_at_unchecked(infinity_len)
                        .0
                }
            };
            let mut values = Values {
                one,
                two,
                four,
                eight,
                half,
                quarter,
                eighth,
            };
            let context = CouplingContext {
                zero,
                infinity,
                split_len: split_width,
                degree,
            };
            Self::evaluate_and_couple_mul(&mut values, temporary, &mut evaluations, a, b, &context);
            Self::interpolate_and_reconstruct(
                initialized,
                split_len,
                values,
                temporary,
                add_mul_kernel,
            );
        }
    }

    /// Square with a balanced eight-way Toom-Cook split.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if !Multiplication::operand_has_eight_parts(a.len()) {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom6);
            return;
        }
        debug_assert!(
            dst.len() >= a.len().saturating_mul(2),
            "Toom-8 squaring output is shorter than the full square"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom8_sqr_scratch_len(a.len()),
            "Toom-8 squaring scratch buffer is undersized"
        );

        let split_width = Self::multiplication_split_len(MulShape::Balanced, a.len(), a.len());
        let split_len = split_width.get();
        // SAFETY: m=ceil(|a|/8) for a real limb slice; its byte bound leaves
        // room for m+ceil(25/LIMB_BITS) and 3m+2*ceil(25/LIMB_BITS), including
        // two evaluation/four interpolation guards on a 16-bit target.
        let (eval_len, packed_len) = unsafe {
            (
                split_len.unchecked_add(EVALUATION_GUARD_BITS.div_ceil(LIMB_BITS)),
                split_len
                    .unchecked_mul(3)
                    .unchecked_add(INTERPOLATION_GUARD_LIMBS),
            )
        };
        let place_points = Self::destination_points_fit(dst.len(), split_len, packed_len, 0);
        let fast_paired_add_sub = ArchKernels::fast_add_sub_limbs_available();
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();
        let (points, temporary, mut evaluations) = Self::split_sqr_scratch(
            scratch,
            packed_len,
            eval_len,
            place_points,
            fast_paired_add_sub,
            add_mul_kernel,
        );
        // SAFETY: eight-part admission gives |a|>7m, so its low m-limb part
        // exists and its 2m-limb square fits the validated complete destination.
        let (low, zero_len, zero_product) = unsafe {
            let low = a.split_at_unchecked(split_len).0;
            let zero_len = split_len.unchecked_mul(2);
            (low, zero_len, dst.split_at_mut_unchecked(zero_len).0)
        };
        Recursive::recursive_sqr(zero_product, low, evaluations.scratch, TierCeiling::Toom6);
        Self::clear_destination_gaps(dst, split_len, packed_len, zero_len, 0, place_points);

        let BasePoints {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
        } = points;
        if place_points {
            let placed = Self::split_destination_points(dst, split_len, packed_len, 0);
            let mut values = Values {
                one: placed.one,
                two: &mut *two,
                four: placed.four,
                eight: &mut *eight,
                half: placed.half,
                quarter: &mut *quarter,
                eighth: &mut *eighth,
            };
            let context = CouplingContext {
                zero: placed.zero,
                infinity: &[],
                split_len: split_width,
                degree: 14,
            };
            Self::evaluate_and_couple_sqr(&mut values, temporary, &mut evaluations, a, &context);
            Self::interpolate_values(values, temporary, add_mul_kernel);
            Self::reconstruct_alternating(
                dst,
                split_len,
                [eighth, quarter, two, eight],
                fast_paired_add_sub,
            );
        } else {
            // SAFETY: zero_len=2m is the initialized endpoint prefix.
            let (zero, _) = unsafe { dst.split_at_unchecked(zero_len) };
            let mut values = Values {
                one,
                two,
                four,
                eight,
                half,
                quarter,
                eighth,
            };
            let context = CouplingContext {
                zero,
                infinity: &[],
                split_len: split_width,
                degree: 14,
            };
            Self::evaluate_and_couple_sqr(&mut values, temporary, &mut evaluations, a, &context);
            Self::interpolate_and_reconstruct(dst, split_len, values, temporary, add_mul_kernel);
        }
    }

    /// Toom-8 evaluation proves each guard is below `2^EVALUATION_GUARD_BITS`, but
    /// that does not fit a 16-bit limb, so the shared bound assertion is left open
    /// here and the guard product is always given its two-limb width.
    const GUARD_BOUND: Limb = Limb::MAX;

    pub fn mul_evaluation(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        split_len: usize,
        kernel: AddMulKernel,
    ) {
        // At short widths, retaining an m-by-m recursive product avoids a tier
        // size discontinuity. The generated crossover records where the four
        // linear guard passes become dearer than the complete (m+1)-limb product.
        // SAFETY: every evaluation has at least m+1 limbs, bounding this sum
        // by its actual slice length even on the two-guard 16-bit layout.
        let guarded_len = unsafe { split_len.unchecked_add(1) };
        if split_len < TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS
            && a.len() == guarded_len
            && b.len() == guarded_len
        {
            Recursive::guarded_evaluation_product::<{ Self::GUARD_BOUND }, 2, _>(
                dst,
                a,
                b,
                scratch,
                kernel,
                |p, low_a, low_b, s| {
                    Recursive::recursive_mul(p, low_a, low_b, s, TierCeiling::Toom6);
                },
            );
            return;
        }
        let active_a = Self::evaluation_prefix(a, split_len);
        let active_b = Self::evaluation_prefix(b, split_len);
        Recursive::recursive_mul(dst, active_a, active_b, scratch, TierCeiling::Toom6);
    }

    pub fn sqr_evaluation(
        dst: &mut [Limb],
        value: &[Limb],
        scratch: &mut [Limb],
        split_len: usize,
        kernel: AddMulKernel,
    ) {
        // SAFETY: evaluation allocates at least m+1 limbs on every target.
        let guarded_len = unsafe { split_len.unchecked_add(1) };
        if value.len() == guarded_len {
            Recursive::guarded_evaluation_square::<{ Self::GUARD_BOUND }, 2>(
                dst,
                value,
                scratch,
                kernel,
                |square, low, s| {
                    Recursive::recursive_sqr(square, low, s, TierCeiling::Toom6);
                },
            );
            return;
        }
        let active = Self::evaluation_prefix(value, split_len);
        Recursive::recursive_sqr(dst, active, scratch, TierCeiling::Toom6);
    }
    /// Removes only inactive guard limbs, retaining the planned low-block width.
    ///
    /// Cancellation can make either low block arbitrarily sparse. Keeping that
    /// block zero-extended prevents operand contents from selecting an unplanned
    /// unequal child with a different scratch layout. At most two guard limbs are
    /// inspected on the supported pointer widths.
    fn evaluation_prefix(value: &[Limb], split_len: usize) -> &[Limb] {
        // SAFETY: every evaluated value contains m initialized body limbs and
        // at least one initialized guard; the caller supplies split_len=m.
        let guard = unsafe { value.split_at_unchecked(split_len).1 };
        // SAFETY: active_len(guard) <= guard.len() == value.len() - split_len, so
        // the sum is bounded by the valid input slice's length on every target.
        let active_len = unsafe { split_len.unchecked_add(SharedEval::active_len(guard)) };
        // SAFETY: active_len=m+active_len(guard)<=value.len(); this keeps all
        // body limbs even when their magnitude is zero, and trims only guard zeros.
        unsafe { value.split_at_unchecked(active_len).0 }
    }
}
