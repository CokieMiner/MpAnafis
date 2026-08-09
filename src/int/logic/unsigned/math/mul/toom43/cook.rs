//! The Toom-Cook 4-by-3 driver: split, evaluate, recurse, interpolate.

use super::{
    AddMulKernel, ArchKernels, Limb, MiddleCoefficients, MiddleProducts, Multiplication, Recursive,
    SharedEval, TierCeiling, Widths,
};

/// Namespace for the four-by-three Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom43;

impl Toom43 {
    /// Multiply a four-part operand by a three-part one.
    ///
    /// Total for the shape [`Widths::toom43_suitable`] admits, which is the only
    /// shape the selector names this tier for. As in the three-by-two split there is
    /// no internal re-dispatch, so the scratch sizing has no fallback to mirror.
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
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
        let eval_len = split_len.wrapping_add(1);
        // `toom43_suitable` proved `larger > 3*split_len` and `2*split_len < smaller
        // <= 3*split_len`, so both high parts are nonempty and neither operand
        // overflows its part count.
        let (part0_a, after_part0_a) = larger.split_at(split_len);
        let (part1_a, after_part1_a) = after_part0_a.split_at(split_len);
        let (part2_a, part3_a) = after_part1_a.split_at(split_len);
        let (part0_b, after_part0_b) = smaller.split_at(split_len);
        let (part1_b, part2_b) = after_part0_b.split_at(split_len);

        let ScratchLayout {
            one,
            negative_one,
            two,
            negative_two,
            evaluation_a,
            evaluation_b,
            negative_evaluation_a,
            negative_evaluation_b,
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
                negative_evaluation_a,
                negative_evaluation_b,
                inner,
            },
            add_mul_kernel,
        );

        // W(0) and W(inf) are exact coefficients and go straight to their radix
        // positions. The destination is exactly `5*split_len + |a3| + |b2|` limbs,
        // so the infinity product ends flush with the end of the product.
        let low_product_len = part0_a.len().wrapping_add(part0_b.len());
        let high_product_len = part3_a.len().wrapping_add(part2_b.len());
        let high_offset = split_len.wrapping_mul(5);
        {
            let (zero_product, _) = dst.split_at_mut(low_product_len);
            Recursive::recursive_mul(zero_product, part0_a, part0_b, inner, TierCeiling::Full);
        }
        {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (infinity_product, _) = high_and_after.split_at_mut(high_product_len);
            Recursive::recursive_mul(infinity_product, part3_a, part2_b, inner, TierCeiling::Full);
        }
        SharedEval::clear_destination_between_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );

        let MiddleCoefficients {
            linear,
            quadratic,
            cubic,
            quartic,
        } = {
            let (before_high, high_and_after) = dst.split_at(high_offset);
            let (zero_product, _) = before_high.split_at(low_product_len);
            let (infinity_product, _) = high_and_after.split_at(high_product_len);
            Self::interpolate_middle(
                MiddleProducts {
                    one,
                    negative_one,
                    two,
                    negative_two,
                    one_is_negative,
                    two_is_negative,
                },
                zero_product,
                infinity_product,
            )
        };
        SharedEval::add_coefficient_in_place(dst, linear, split_len);
        SharedEval::add_coefficient_in_place(dst, quadratic, split_len.wrapping_mul(2));
        SharedEval::add_coefficient_in_place(dst, cubic, split_len.wrapping_mul(3));
        SharedEval::add_coefficient_in_place(dst, quartic, split_len.wrapping_mul(4));
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
    negative_evaluation_a: &'buffer mut [Limb],
    negative_evaluation_b: &'buffer mut [Limb],
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
        negative_evaluation_a,
        negative_evaluation_b,
        inner,
    } = buffers;

    let one_is_negative = evaluate_and_multiply::<false>(
        one,
        negative_one,
        long,
        short,
        evaluation_a,
        evaluation_b,
        negative_evaluation_a,
        negative_evaluation_b,
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
        negative_evaluation_a,
        negative_evaluation_b,
        inner,
        add_mul_kernel,
    );
    (one_is_negative, two_is_negative)
}

/// Evaluate both operands at `+x` and `-x` and run the two products.
#[allow(
    clippy::too_many_arguments,
    reason = "the buffers are already grouped by the caller's structs; regrouping them again here \
              would only rename the same five borrows"
)]
fn evaluate_and_multiply<const AT_TWO: bool>(
    positive_product: &mut [Limb],
    negative_product: &mut [Limb],
    long: [&[Limb]; 4],
    short: [&[Limb]; 3],
    evaluation_a: &mut [Limb],
    evaluation_b: &mut [Limb],
    negative_evaluation_a: &mut [Limb],
    negative_evaluation_b: &mut [Limb],
    inner: &mut [Limb],
    add_mul_kernel: AddMulKernel,
) -> bool {
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
    recursive_evaluation_product(
        positive_product,
        evaluation_a,
        evaluation_b,
        inner,
        add_mul_kernel,
    );
    recursive_evaluation_product(
        negative_product,
        negative_evaluation_a,
        negative_evaluation_b,
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
fn recursive_evaluation_product(
    dst: &mut [Limb],
    evaluation_a: &[Limb],
    evaluation_b: &[Limb],
    scratch: &mut [Limb],
    add_mul_kernel: AddMulKernel,
) {
    Recursive::guarded_evaluation_product::<15, 1>(
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

// ── Scratch layout ───────────────────────────────────────────────────────────

struct ScratchLayout<'buffer> {
    one: &'buffer mut [Limb],
    negative_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    negative_two: &'buffer mut [Limb],
    evaluation_a: &'buffer mut [Limb],
    evaluation_b: &'buffer mut [Limb],
    negative_evaluation_a: &'buffer mut [Limb],
    negative_evaluation_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

const fn split_scratch(scratch: &mut [Limb], eval_len: usize) -> ScratchLayout<'_> {
    let product_len = eval_len.wrapping_mul(2);
    let (one, after_one) = scratch.split_at_mut(product_len);
    let (negative_one, after_negative_one) = after_one.split_at_mut(product_len);
    let (two, after_two) = after_negative_one.split_at_mut(product_len);
    let (negative_two, after_negative_two) = after_two.split_at_mut(product_len);
    let (evaluation_a, after_evaluation_a) = after_negative_two.split_at_mut(eval_len);
    let (evaluation_b, after_evaluation_b) = after_evaluation_a.split_at_mut(eval_len);
    let (negative_evaluation_a, after_negative_a) = after_evaluation_b.split_at_mut(eval_len);
    let (negative_evaluation_b, inner) = after_negative_a.split_at_mut(eval_len);
    ScratchLayout {
        one,
        negative_one,
        two,
        negative_two,
        evaluation_a,
        evaluation_b,
        negative_evaluation_a,
        negative_evaluation_b,
        inner,
    }
}

/// Fixed workspace one Toom-4-by-3 level needs above its recursive children.
///
/// Four middle products at twice the evaluation width, and four evaluation
/// buffers: the positive and negative evaluation of each operand stay live
/// together so a signed pair costs one pass rather than two.
impl Toom43 {
    pub const fn local_scratch_len(split_len: usize, inner_space: usize) -> usize {
        let eval_len = split_len.wrapping_add(1);
        eval_len.wrapping_mul(12).wrapping_add(inner_space)
    }
}
