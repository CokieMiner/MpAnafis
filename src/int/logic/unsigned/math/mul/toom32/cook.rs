//! The Toom-Cook 3-by-2 driver: split, evaluate, recurse, interpolate.

use super::{
    AddMulKernel, ArchKernels, Limb, Multiplication, Recursive, SharedEval, TierCeiling, Toom3,
    Widths,
};

/// Namespace for the three-by-two Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom32;

impl Toom32 {
    /// Multiply a three-part operand by a two-part one.
    ///
    /// Total for the shape [`Widths::toom32_suitable`] admits, which is the only
    /// shape the selector names this tier for. There is deliberately no internal
    /// re-dispatch: a tier that falls back inside its own driver has to be mirrored
    /// by its scratch sizing, and keeping those two in agreement by hand is what
    /// desynchronised Toom-6 from `toom6_mul_scratch_len` when the blocked crossover
    /// moved.
    pub fn mul(dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        let (larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        debug_assert!(
            Widths::new(a.len(), b.len()).toom32_suitable(),
            "Toom-3-by-2 was selected for a shape it cannot split"
        );
        debug_assert!(
            scratch.len() >= Multiplication::toom32_mul_scratch_len(a.len(), b.len()),
            "Toom-3-by-2 scratch is undersized: have {}, need {} for {}x{} limbs",
            scratch.len(),
            Multiplication::toom32_mul_scratch_len(a.len(), b.len()),
            a.len(),
            b.len()
        );

        let split_len = larger.len().div_ceil(3);
        let eval_len = split_len.wrapping_add(1);
        // `toom32_suitable` proved `larger > 2*split_len` and `split_len < smaller
        // <= 2*split_len`, so both splits leave a nonempty high part.
        let (part0_a, after_part0_a) = larger.split_at(split_len);
        let (part1_a, part2_a) = after_part0_a.split_at(split_len);
        let (part0_b, part1_b) = smaller.split_at(split_len);

        let ScratchLayout {
            one,
            negative_one,
            positive_a,
            positive_b,
            negative_a,
            negative_b,
            inner,
        } = split_scratch(scratch, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        // Both signed evaluations are formed before either product runs, so the
        // positive and negative operands live in separate buffers and no evaluation
        // has to be recomputed or copied between the two recursive products.
        let long_side_flipped =
            Toom3::evaluate_one_and_negative_one(positive_a, negative_a, part0_a, part1_a, part2_a);
        let short_side_flipped =
            Self::evaluate_one_and_negative_one(positive_b, negative_b, part0_b, part1_b);

        recursive_evaluation_product(one, positive_a, positive_b, inner, add_mul_kernel);
        recursive_evaluation_product(negative_one, negative_a, negative_b, inner, add_mul_kernel);

        // W(0) and W(inf) are written straight to their final radix positions; they
        // need no interpolation and the destination is their only storage.
        let low_product_len = part0_a.len().wrapping_add(part0_b.len());
        let high_product_len = part2_a.len().wrapping_add(part1_b.len());
        let high_offset = split_len.wrapping_mul(3);
        {
            let (zero_product, _) = dst.split_at_mut(low_product_len);
            Recursive::recursive_mul(zero_product, part0_a, part0_b, inner, TierCeiling::Full);
        }
        {
            let (_, high_and_after) = dst.split_at_mut(high_offset);
            let (infinity_product, _) = high_and_after.split_at_mut(high_product_len);
            Recursive::recursive_mul(infinity_product, part2_a, part1_b, inner, TierCeiling::Full);
        }
        SharedEval::clear_destination_between_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );

        let (linear, quadratic) = {
            let (before_high, high_and_after) = dst.split_at(high_offset);
            let (zero_product, _) = before_high.split_at(low_product_len);
            let (infinity_product, _) = high_and_after.split_at(high_product_len);
            Self::interpolate_middle(
                one,
                negative_one,
                zero_product,
                infinity_product,
                long_side_flipped ^ short_side_flipped,
            )
        };
        SharedEval::add_coefficient_in_place(dst, linear, split_len);
        SharedEval::add_coefficient_in_place(dst, quadratic, split_len.wrapping_mul(2));
    }
}

/// Toom-3-by-2 evaluations carry a guard below three.
///
/// `A(1) = a0 + a1 + a2 < 3*B^m` and `B(1) = b0 + b1 < 2*B^m`, and both negative
/// magnitudes are bounded by their positive counterparts, so no guard limb
/// reaches three. Splitting the guard out keeps the recursive product at exactly
/// `split_len` limbs per side, which is what lets the child land on a balanced
/// tier rather than a one-limb-lopsided one.
fn recursive_evaluation_product(
    dst: &mut [Limb],
    evaluation_a: &[Limb],
    evaluation_b: &[Limb],
    scratch: &mut [Limb],
    add_mul_kernel: AddMulKernel,
) {
    Recursive::guarded_evaluation_product::<3, 1>(
        dst,
        evaluation_a,
        evaluation_b,
        scratch,
        add_mul_kernel,
        |product, low_a, low_b, inner| {
            Recursive::recursive_mul(product, low_a, low_b, inner, TierCeiling::Full);
        },
    );
}

// ── Scratch layout ───────────────────────────────────────────────────────────

struct ScratchLayout<'buffer> {
    one: &'buffer mut [Limb],
    negative_one: &'buffer mut [Limb],
    positive_a: &'buffer mut [Limb],
    positive_b: &'buffer mut [Limb],
    negative_a: &'buffer mut [Limb],
    negative_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

const fn split_scratch(scratch: &mut [Limb], eval_len: usize) -> ScratchLayout<'_> {
    let product_len = eval_len.wrapping_mul(2);
    let (one, after_one) = scratch.split_at_mut(product_len);
    let (negative_one, after_negative_one) = after_one.split_at_mut(product_len);
    let (positive_a, after_positive_a) = after_negative_one.split_at_mut(eval_len);
    let (positive_b, after_positive_b) = after_positive_a.split_at_mut(eval_len);
    let (negative_a, after_negative_a) = after_positive_b.split_at_mut(eval_len);
    let (negative_b, inner) = after_negative_a.split_at_mut(eval_len);
    ScratchLayout {
        one,
        negative_one,
        positive_a,
        positive_b,
        negative_a,
        negative_b,
        inner,
    }
}

/// Fixed workspace one Toom-3-by-2 level needs above its recursive children.
///
/// Two middle products at twice the evaluation width, and four evaluation
/// buffers: unlike the balanced tiers this keeps the positive and negative
/// evaluations of *both* operands live simultaneously, which trades two
/// evaluation buffers for the copy each balanced tier makes between its two
/// point products.
impl Toom32 {
    pub const fn local_scratch_len(split_len: usize, inner_space: usize) -> usize {
        let eval_len = split_len.wrapping_add(1);
        eval_len.wrapping_mul(8).wrapping_add(inner_space)
    }
}
