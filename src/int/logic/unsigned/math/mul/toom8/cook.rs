//! The Toom-Cook 8 and 8.5 drivers: split, couple, interpolate, reconstruct.
use core::cmp::min;

use super::{
    AddMulKernel, ArchKernels, CouplingContext, Limb, MulEvaluationBuffers, MulScratchLayout,
    MulShape, Multiplication, Recursive, SharedEval, SqrEvaluationBuffers, SqrScratchLayout,
    TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, TierCeiling, Values, Widths,
};

pub const BALANCED_PARTS: usize = 8;
pub const HALF_LARGE_PARTS: usize = 9;
pub const HALF_SMALL_PARTS: usize = 8;
pub const EVALUATION_GUARD_BITS: usize = 25;
pub const INTERPOLATION_GUARD_BITS: usize = 96;

/// Namespace for the eight-way and eight-and-a-half-way Toom-Cook tiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom8;

pub struct ProductPair<'buffer> {
    pub positive: &'buffer mut [Limb],
    pub negative: &'buffer mut [Limb],
}

impl Toom8 {
    /// Multiply with a balanced Toom-8 or adjacent unbalanced Toom-8.5 split.
    #[allow(
        clippy::too_many_lines,
        reason = "The driver keeps one linear split-evaluate-interpolate sequence whose buffers and lifetimes are coupled"
    )]
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let Some(shape) = Widths::new(a.len(), b.len()).toom8_shape() else {
            Recursive::recursive_mul(dst, a, b, scratch, TierCeiling::Toom6);
            return;
        };
        debug_assert!(
            dst.len() >= a.len().wrapping_add(b.len()),
            "Toom-8 multiplication output is shorter than the full product"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom8_mul_scratch_len(a.len(), b.len()),
            "Toom-8 multiplication scratch buffer is undersized"
        );

        let split_len = Self::multiplication_split_len(shape, a.len(), b.len());
        let degree = Self::multiplication_degree(shape);
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let packed_len = value_len.wrapping_add(split_len);
        let expected_infinity_len =
            Self::multiplication_infinity_len(shape, split_len, a.len(), b.len());
        let place_points =
            Self::destination_points_fit(dst.len(), split_len, packed_len, expected_infinity_len);
        let layout = Self::split_mul_scratch(scratch, split_len, value_len, eval_len, place_points);
        let (zero_product_len, infinity_len) =
            Self::multiply_endpoints(dst, a, b, layout.inner, shape, split_len);
        debug_assert_eq!(
            infinity_len, expected_infinity_len,
            "Toom-8.5 endpoint geometry disagrees with scratch sizing"
        );
        let MulScratchLayout {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
            temporary,
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            inner,
        } = layout;
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();
        let mut evaluations = MulEvaluationBuffers {
            eval_a,
            eval_b,
            odd_a,
            odd_b,
            scratch: inner,
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
            add_mul_kernel,
        };
        Self::clear_destination_gaps(
            dst,
            split_len,
            packed_len,
            zero_product_len,
            infinity_len,
            place_points,
        );
        if let Some(placed) =
            Self::split_destination_points(dst, split_len, packed_len, infinity_len)
        {
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
                infinity: placed.infinity,
                split_len,
                degree,
            };
            Self::evaluate_and_couple_mul(&mut values, temporary, &mut evaluations, a, b, &context);
            Self::interpolate_values(values, temporary, add_mul_kernel);
            Self::reconstruct_alternating(
                dst,
                split_len,
                [eighth, quarter, two, eight],
                evaluations.fast_paired_add_sub,
            );
        } else {
            let (zero, infinity) = Self::endpoint_slices(dst, split_len, infinity_len);
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
                split_len,
                degree,
            };
            Self::evaluate_and_couple_mul(&mut values, temporary, &mut evaluations, a, b, &context);
            Self::interpolate_and_reconstruct(dst, split_len, values, temporary, add_mul_kernel);
        }
    }

    /// Square with a balanced eight-way Toom-Cook split.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if !Multiplication::operand_has_eight_parts(a.len()) {
            Recursive::recursive_sqr(dst, a, scratch, TierCeiling::Toom6);
            return;
        }
        debug_assert!(
            dst.len() >= a.len().wrapping_mul(2),
            "Toom-8 squaring output is shorter than the full square"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom8_sqr_scratch_len(a.len()),
            "Toom-8 squaring scratch buffer is undersized"
        );

        let split_len = a.len().div_ceil(BALANCED_PARTS);
        let eval_len = Self::evaluation_len(split_len);
        let value_len = Self::interpolation_value_len(split_len);
        let packed_len = value_len.wrapping_add(split_len);
        let place_points = Self::destination_points_fit(dst.len(), split_len, packed_len, 0);
        let layout = Self::split_sqr_scratch(scratch, split_len, value_len, eval_len, place_points);
        let low_len = min(a.len(), split_len);
        let (low, _) = a.split_at(low_len);
        let zero_len = low.len().wrapping_mul(2);
        let (zero_product, _) = dst.split_at_mut(zero_len);
        Recursive::recursive_sqr(zero_product, low, layout.inner, TierCeiling::Toom6);
        Self::clear_destination_gaps(dst, split_len, packed_len, zero_len, 0, place_points);

        let SqrScratchLayout {
            one,
            two,
            four,
            eight,
            half,
            quarter,
            eighth,
            temporary,
            eval,
            odd,
            inner,
        } = layout;
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();
        let mut evaluations = SqrEvaluationBuffers {
            eval,
            odd,
            scratch: inner,
            fast_paired_add_sub: ArchKernels::fast_add_sub_limbs_available(),
            add_mul_kernel,
        };
        if let Some(placed) = Self::split_destination_points(dst, split_len, packed_len, 0) {
            let mut values = Values {
                one: placed.one,
                two,
                four: placed.four,
                eight,
                half: placed.half,
                quarter,
                eighth,
            };
            let context = CouplingContext {
                zero: placed.zero,
                infinity: &[],
                split_len,
                degree: Self::multiplication_degree(MulShape::Balanced),
            };
            Self::evaluate_and_couple_sqr(&mut values, temporary, &mut evaluations, a, &context);
            Self::interpolate_values(values, temporary, add_mul_kernel);
            Self::reconstruct_alternating(
                dst,
                split_len,
                [eighth, quarter, two, eight],
                evaluations.fast_paired_add_sub,
            );
        } else {
            let (zero, _) = dst.split_at(split_len.wrapping_mul(2));
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
                split_len,
                degree: Self::multiplication_degree(MulShape::Balanced),
            };
            Self::evaluate_and_couple_sqr(&mut values, temporary, &mut evaluations, a, &context);
            Self::interpolate_and_reconstruct(dst, split_len, values, temporary, add_mul_kernel);
        }
    }

    // ── Recursive point products and guard expansion ──────────────────────────────

    /// Toom-8 evaluation proves each guard is below `2^EVALUATION_GUARD_BITS`, but
    /// that does not fit a 16-bit limb, so the shared bound assertion is left open
    /// here and the guard product is always given its two-limb width.
    const GUARD_BOUND: Limb = Limb::MAX;

    pub fn multiply_active(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        split_len: usize,
        kernel: AddMulKernel,
    ) {
        // At short widths, retaining an m-by-m recursive product avoids a tier
        // size discontinuity. The generated crossover records where the four
        // linear guard passes become dearer than the complete (m+1)-limb product.
        let guarded_len = split_len.wrapping_add(1);
        if split_len < TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS
            && a.len() == guarded_len
            && b.len() == guarded_len
        {
            expand_guard_product(dst, a, b, scratch, kernel);
            return;
        }
        let active_a = active_prefix(a);
        let active_b = active_prefix(b);
        Recursive::recursive_mul(dst, active_a, active_b, scratch, TierCeiling::Toom6);
    }

    pub fn square_active(
        dst: &mut [Limb],
        value: &[Limb],
        scratch: &mut [Limb],
        split_len: usize,
        kernel: AddMulKernel,
    ) {
        if value.len() == split_len.wrapping_add(1) {
            expand_guard_square(dst, value, scratch, kernel);
            return;
        }
        let active = active_prefix(value);
        let product_len = active.len().wrapping_mul(2);
        let (product, guard) = dst.split_at_mut(product_len);
        guard.fill(0);
        Recursive::recursive_sqr(product, active, scratch, TierCeiling::Toom6);
    }
}

fn expand_guard_product(
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    scratch: &mut [Limb],
    kernel: AddMulKernel,
) {
    Recursive::guarded_evaluation_product::<{ Toom8::GUARD_BOUND }, 2>(
        dst,
        a,
        b,
        scratch,
        kernel,
        |p, low_a, low_b, s| {
            Recursive::recursive_mul(p, low_a, low_b, s, TierCeiling::Toom6);
        },
    );
}

fn expand_guard_square(
    dst: &mut [Limb],
    value: &[Limb],
    scratch: &mut [Limb],
    kernel: AddMulKernel,
) {
    Recursive::guarded_evaluation_square::<{ Toom8::GUARD_BOUND }, 2>(
        dst,
        value,
        scratch,
        kernel,
        |square, low, s| {
            Recursive::recursive_sqr(square, low, s, TierCeiling::Toom6);
        },
    );
}

fn active_prefix(value: &[Limb]) -> &[Limb] {
    let (active, _) = value.split_at(SharedEval::active_len(value));
    active
}
