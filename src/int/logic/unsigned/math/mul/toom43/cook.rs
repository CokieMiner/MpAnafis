//! The Toom-Cook 4-by-3 driver: split, evaluate, recurse, interpolate.
//!
//! # References
//!
//! - Bodrato, M., & Zanoni, A. (2007). Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices. *Proceedings of ISSAC '07*, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.

#![expect(
    unsafe_code,
    reason = "Shape admission and checked scratch sizing establish all endpoint and workspace partition bounds"
)]

use super::{
    AddMulKernel, ArchKernels, Limb, LimbOutput, MiddleCoefficients, MiddleProducts,
    Multiplication, Recursive, SharedEval, TierCeiling, Widths,
};

/// Namespace for the four-by-three Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom43;

impl Toom43 {
    /// Multiply a four-part operand by a three-part one.
    ///
    /// Requires [`Widths::toom43_suitable`], a complete destination, and
    /// [`Multiplication::toom43_mul_scratch_len`] scratch limbs.
    pub fn mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        debug_assert!(
            Widths::new(a.len(), b.len()).toom43_suitable(),
            "Toom-4-by-3 was selected for a shape it cannot split"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom43_mul_scratch_len(a.len(), b.len()),
            "Toom-4-by-3 scratch is undersized: have {}, need {} for {}x{} limbs",
            scratch.len(),
            Multiplication::toom43_mul_scratch_len(a.len(), b.len()),
            a.len(),
            b.len()
        );

        let split_len = larger.len().div_ceil(4);
        // SAFETY: m=ceil(larger.len()/4) for a real limb slice, whose byte
        // length is <=isize::MAX; m+1 fits all supported pointer widths.
        let eval_len = unsafe { split_len.unchecked_add(1) };
        // `toom43_suitable` proved `larger > 3*split_len` and `2*split_len < smaller
        // <= 3*split_len`, so both high parts are nonempty and neither operand
        // overflows its part count.
        // SAFETY: admission gives |larger|>3m and 2m<|smaller|<=3m.
        // Consecutive immutable splits preserve full low blocks and nonempty endpoints.
        let (part0_a, part1_a, part2_a, part3_a, part0_b, part1_b, part2_b) = unsafe {
            let (part0_a, after_part0_a) = larger.split_at_unchecked(split_len);
            let (part1_a, after_part1_a) = after_part0_a.split_at_unchecked(split_len);
            let (part2_a, part3_a) = after_part1_a.split_at_unchecked(split_len);
            let (part0_b, after_part0_b) = smaller.split_at_unchecked(split_len);
            let (part1_b, part2_b) = after_part0_b.split_at_unchecked(split_len);
            (
                part0_a, part1_a, part2_a, part3_a, part0_b, part1_b, part2_b,
            )
        };

        let ScratchLayout {
            one,
            negative_one,
            two,
            negative_two,
            evaluation_a,
            evaluation_b,
            inner,
        } = split_scratch(scratch, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        let (one_is_negative, two_is_negative) = multiply_points(
            PointProducts {
                one,
                negative_one,
                two,
                negative_two,
            },
            &OperandParts {
                long: [part0_a, part1_a, part2_a, part3_a],
                short: [part0_b, part1_b, part2_b],
            },
            PointBuffers {
                evaluation_a,
                evaluation_b,
                inner,
            },
            add_mul_kernel,
        );

        // W(0) and W(inf) are exact coefficients and go straight to their radix
        // positions. The destination is exactly `5*split_len + |a3| + |b2|` limbs,
        // so the infinity product ends flush with the end of the product.
        // SAFETY: full low blocks give a 2m-limb zero product. Nonempty high
        // blocks make the exact full product 5m+|a3|+|b2|<=dst.len() limbs.
        let (low_product_len, high_product_len, high_offset) = unsafe {
            (
                split_len.unchecked_mul(2),
                part3_a.len().unchecked_add(part2_b.len()),
                split_len.unchecked_mul(5),
            )
        };
        // SAFETY: the endpoints [0,2m) and [5m,|a|+|b|) are disjoint,
        // contained in dst. Partition once, retaining them through interpolation;
        // only the intervening canvas and optional product suffix need zeroing.
        let (zero_product, middle_gap, infinity_product, trailing_gap) = unsafe {
            let (before_high, high_and_after) = dst.split_at_mut_unchecked(high_offset);
            let (zero, middle) = before_high.split_at_mut_unchecked(low_product_len);
            let (infinity, tail) = high_and_after.split_at_mut_unchecked(high_product_len);
            (zero, middle, infinity, tail)
        };
        middle_gap.fill(LimbOutput::from_limb(0));
        trailing_gap.fill(LimbOutput::from_limb(0));
        Recursive::recursive_mul(zero_product, part0_a, part0_b, inner, TierCeiling::Full);
        Recursive::recursive_mul(infinity_product, part3_a, part2_b, inner, TierCeiling::Full);
        // SAFETY: the endpoint products initialized their entire disjoint
        // spans before the interpolation reads either exact coefficient.
        let (zero_value, infinity_value) = unsafe {
            (
                LimbOutput::assume_init_mut(zero_product),
                LimbOutput::assume_init_mut(infinity_product),
            )
        };

        let MiddleCoefficients {
            linear,
            quadratic,
            cubic,
            quartic,
        } = Self::interpolate_middle(
            MiddleProducts {
                one,
                negative_one,
                two,
                negative_two,
                one_is_negative,
                two_is_negative,
            },
            zero_value,
            infinity_value,
        );
        // SAFETY: endpoint writers and the intervening/trailing gap fills
        // together initialized every element of the complete destination.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };
        SharedEval::add_coefficient_in_place(initialized, linear, split_len);
        SharedEval::add_coefficient_in_place(initialized, quadratic, low_product_len);
        // SAFETY: these coefficient shifts precede the in-bounds 5m endpoint.
        let (cubic_offset, quartic_offset) =
            unsafe { (split_len.unchecked_mul(3), split_len.unchecked_mul(4)) };
        SharedEval::add_coefficient_in_place(initialized, cubic, cubic_offset);
        SharedEval::add_coefficient_in_place(initialized, quartic, quartic_offset);
    }
}

/// The four polynomial parts of the longer operand and three of the shorter.
#[derive(Clone, Copy)]
struct OperandParts<'value> {
    long: [&'value [Limb]; 4],
    short: [&'value [Limb]; 3],
}

/// Destinations for the four signed point products.
struct PointProducts<'buffer> {
    one: &'buffer mut [Limb],
    negative_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    negative_two: &'buffer mut [Limb],
}

/// Reusable evaluation and recursive-work buffers.
struct PointBuffers<'buffer> {
    evaluation_a: &'buffer mut [Limb],
    evaluation_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

/// Evaluate and multiply all four signed points, returning the two signs.
///
/// Both points of a pair come from one pass over the parts, so the four
/// recursive products cost two evaluation passes per operand rather than four.
/// The `x = 1` pair is fully consumed before the `x = 2` pair overwrites its
/// evaluation buffers.
fn multiply_points(
    products: PointProducts<'_>,
    parts: &OperandParts<'_>,
    buffers: PointBuffers<'_>,
    add_mul_kernel: AddMulKernel,
) -> (bool, bool) {
    let PointProducts {
        one,
        negative_one,
        two,
        negative_two,
    } = products;
    let OperandParts { long, short } = *parts;
    let PointBuffers {
        evaluation_a,
        evaluation_b,
        inner,
    } = buffers;

    let one_is_negative = evaluate_and_multiply::<false>(
        one,
        negative_one,
        long,
        short,
        evaluation_a,
        evaluation_b,
        inner,
        add_mul_kernel,
    );
    let two_is_negative = evaluate_and_multiply::<true>(
        two,
        negative_two,
        long,
        short,
        evaluation_a,
        evaluation_b,
        inner,
        add_mul_kernel,
    );
    (one_is_negative, two_is_negative)
}

/// Evaluate both operands at `+x` and `-x` and run the two products.
#[expect(
    clippy::too_many_arguments,
    reason = "the buffers are already grouped by the caller's structs; regrouping them again here \
              would only rename the same three borrows"
)]
fn evaluate_and_multiply<const AT_TWO: bool>(
    positive_product: &mut [Limb],
    negative_product: &mut [Limb],
    long: [&[Limb]; 4],
    short: [&[Limb]; 3],
    evaluation_a: &mut [Limb],
    evaluation_b: &mut [Limb],
    inner: &mut [Limb],
    add_mul_kernel: AddMulKernel,
) -> bool {
    // The negative evaluations die before the positive product is written.
    // Its two evaluation-width halves therefore need no separate allocation.
    // SAFETY: the product slot spans two equal evaluation widths, so this
    // split exposes disjoint temporary negative evaluations of that same width.
    let (negative_evaluation_a, negative_evaluation_b) =
        unsafe { positive_product.split_at_mut_unchecked(evaluation_a.len()) };
    let flipped_long = Toom43::evaluate_four_parts::<AT_TWO>(
        evaluation_a,
        negative_evaluation_a,
        long[0],
        long[1],
        long[2],
        long[3],
        add_mul_kernel,
    );
    let flipped_short = Toom43::evaluate_three_parts::<AT_TWO>(
        evaluation_b,
        negative_evaluation_b,
        short[0],
        short[1],
        short[2],
        add_mul_kernel,
    );
    mul_evaluation(
        negative_product,
        negative_evaluation_a,
        negative_evaluation_b,
        inner,
        add_mul_kernel,
    );
    mul_evaluation(
        positive_product,
        evaluation_a,
        evaluation_b,
        inner,
        add_mul_kernel,
    );
    flipped_long ^ flipped_short
}

/// Toom-4-by-3 evaluations carry a guard below fifteen.
///
/// The widest is `A(2) = a0 + 2*a1 + 4*a2 + 8*a3 < 15*B^m`; `B(2) < 7*B^m`,
/// `A(1) < 4*B^m`, and `B(1) < 3*B^m` are all smaller, and each negative point
/// is bounded by its positive counterpart. Fifteen squared is 225, so the guard
/// product still occupies a single limb on every supported target.
fn mul_evaluation(
    dst: &mut [Limb],
    evaluation_a: &[Limb],
    evaluation_b: &[Limb],
    scratch: &mut [Limb],
    add_mul_kernel: AddMulKernel,
) {
    Recursive::guarded_evaluation_product::<15, 1, _>(
        dst,
        evaluation_a,
        evaluation_b,
        scratch,
        add_mul_kernel,
        |product, low_a, low_b, recursive| {
            Recursive::recursive_mul(product, low_a, low_b, recursive, TierCeiling::Full);
        },
    );
}

struct ScratchLayout<'buffer> {
    one: &'buffer mut [Limb],
    negative_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    negative_two: &'buffer mut [Limb],
    evaluation_a: &'buffer mut [Limb],
    evaluation_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

const fn split_scratch(scratch: &mut [Limb], eval_len: usize) -> ScratchLayout<'_> {
    // SAFETY: checked local sizing reserves 10*eval_len plus child scratch:
    // four 2*eval_len products and two eval_len evaluations. Every successive
    // split remains in the initialized slice and produces disjoint spans.
    let (one, negative_one, two, negative_two, evaluation_a, evaluation_b, inner) = unsafe {
        let product_len = eval_len.unchecked_mul(2);
        let (one, after_one) = scratch.split_at_mut_unchecked(product_len);
        let (negative_one, after_negative_one) = after_one.split_at_mut_unchecked(product_len);
        let (two, after_two) = after_negative_one.split_at_mut_unchecked(product_len);
        let (negative_two, after_negative_two) = after_two.split_at_mut_unchecked(product_len);
        let (evaluation_a, after_evaluation_a) =
            after_negative_two.split_at_mut_unchecked(eval_len);
        let (evaluation_b, inner) = after_evaluation_a.split_at_mut_unchecked(eval_len);
        (
            one,
            negative_one,
            two,
            negative_two,
            evaluation_a,
            evaluation_b,
            inner,
        )
    };
    ScratchLayout {
        one,
        negative_one,
        two,
        negative_two,
        evaluation_a,
        evaluation_b,
        inner,
    }
}
