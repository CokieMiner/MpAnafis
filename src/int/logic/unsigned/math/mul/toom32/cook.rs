//! The Toom-Cook 3-by-2 driver: split, evaluate, recurse, interpolate.
//!
//! # References
//!
//! - Bodrato, M., & Zanoni, A. (2007). Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices. *Proceedings of ISSAC '07*, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.

#![expect(
    unsafe_code,
    reason = "Three-by-two admission proves nonempty parts, guarded evaluation widths, and disjoint endpoint spans"
)]

use super::{
    AddMulKernel, ArchKernels, Limb, LimbOutput, Multiplication, Recursive, SharedEval,
    TierCeiling, Toom3, Widths,
};

/// Namespace for the three-by-two Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom32;

struct ScratchLayout<'buffer> {
    one: &'buffer mut [Limb],
    negative_one: &'buffer mut [Limb],
    positive_a: &'buffer mut [Limb],
    positive_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

impl Toom32 {
    /// Multiply a three-part operand by a two-part one.
    ///
    /// Requires [`Widths::toom32_suitable`], a complete destination, and
    /// [`Multiplication::toom32_mul_scratch_len`] scratch limbs.
    pub fn mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
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
        // SAFETY: m=ceil(larger.len()/3) and real limb slices occupy at most
        // isize::MAX bytes with >=2 bytes per limb, so m+1 fits every usize.
        let eval_len = unsafe { split_len.unchecked_add(1) };
        // `toom32_suitable` proved `larger > 2*split_len` and `split_len < smaller
        // <= 2*split_len`, so both splits leave a nonempty high part.
        // SAFETY: admission proves |larger|>2m and m<|smaller|<=2m.
        // These immutable splits retain full low blocks and nonempty high blocks.
        let (part0_a, part1_a, part2_a, part0_b, part1_b) = unsafe {
            let (part0_a, after_part0_a) = larger.split_at_unchecked(split_len);
            let (part1_a, part2_a) = after_part0_a.split_at_unchecked(split_len);
            let (part0_b, part1_b) = smaller.split_at_unchecked(split_len);
            (part0_a, part1_a, part2_a, part0_b, part1_b)
        };

        let ScratchLayout {
            one,
            negative_one,
            positive_a,
            positive_b,
            inner,
        } = Self::split_scratch(scratch, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        // The positive-product slot holds both negative evaluations until the
        // negative product consumes them. The positive product then overwrites
        // that dead storage, with no evaluation copies or additional buffers.
        // SAFETY: this product slot spans exactly two evaluation widths.
        let (negative_a, negative_b) = unsafe { one.split_at_mut_unchecked(eval_len) };
        let long_side_flipped =
            Toom3::evaluate_one_and_negative_one(positive_a, negative_a, part0_a, part1_a, part2_a);
        let short_side_flipped =
            Self::evaluate_one_and_negative_one(positive_b, negative_b, part0_b, part1_b);

        Self::mul_evaluation(negative_one, negative_a, negative_b, inner, add_mul_kernel);
        Self::mul_evaluation(one, positive_a, positive_b, inner, add_mul_kernel);

        // W(0) and W(inf) are written straight to their final radix positions; they
        // need no interpolation and the destination is their only storage.
        // SAFETY: low blocks are m limbs and both high blocks are nonempty.
        // The complete product has 3m+|a2|+|b1| limbs, bounded by dst.len().
        let (low_product_len, high_product_len, high_offset) = unsafe {
            (
                split_len.unchecked_mul(2),
                part2_a.len().unchecked_add(part1_b.len()),
                split_len.unchecked_mul(3),
            )
        };
        // SAFETY: [0,2m) and [3m,|a|+|b|) are the exact disjoint endpoints
        // inside dst. Partition once and retain them through interpolation;
        // the intervening m-limb canvas and optional suffix are the only gaps.
        let (zero_product, middle_gap, infinity_product, trailing_gap) = unsafe {
            let (before_high, high_and_after) = dst.split_at_mut_unchecked(high_offset);
            let (zero, middle) = before_high.split_at_mut_unchecked(low_product_len);
            let (infinity, tail) = high_and_after.split_at_mut_unchecked(high_product_len);
            (zero, middle, infinity, tail)
        };
        middle_gap.fill(LimbOutput::from_limb(0));
        trailing_gap.fill(LimbOutput::from_limb(0));
        Recursive::recursive_mul(zero_product, part0_a, part0_b, inner, TierCeiling::Full);
        Recursive::recursive_mul(infinity_product, part2_a, part1_b, inner, TierCeiling::Full);
        // SAFETY: each endpoint writer initialized its complete exact span;
        // these borrows read only the endpoints, independently of the canvas.
        let (zero_value, infinity_value) = unsafe {
            (
                LimbOutput::assume_init_mut(zero_product),
                LimbOutput::assume_init_mut(infinity_product),
            )
        };

        let (linear, quadratic) = Self::interpolate_middle(
            one,
            negative_one,
            zero_value,
            infinity_value,
            long_side_flipped ^ short_side_flipped,
        );
        // SAFETY: the two endpoints and the two zeroed gaps partition and
        // initialize the entire destination before overlapping reconstruction.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };
        SharedEval::add_coefficient_in_place(initialized, linear, split_len);
        SharedEval::add_coefficient_in_place(initialized, quadratic, low_product_len);
    }
    /// Evaluated magnitudes are below `3*B^m`; recurse on their m-limb bodies.
    fn mul_evaluation(
        dst: &mut [Limb],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_product::<3, 1, _>(
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

    /// Partitions two full evaluated products, two operands, and child scratch.
    const fn split_scratch(scratch: &mut [Limb], eval_len: usize) -> ScratchLayout<'_> {
        // SAFETY: dispatch sizing validated 6*eval_len plus child scratch.
        // Two products each take 2*eval_len and two evaluations take eval_len;
        // linear splits retain initialized, disjoint storage without changing length.
        let (one, negative_one, positive_a, positive_b, inner) = unsafe {
            let product_len = eval_len.unchecked_mul(2);
            let (one, after_one) = scratch.split_at_mut_unchecked(product_len);
            let (negative_one, after_negative_one) = after_one.split_at_mut_unchecked(product_len);
            let (positive_a, after_positive_a) =
                after_negative_one.split_at_mut_unchecked(eval_len);
            let (positive_b, inner) = after_positive_a.split_at_mut_unchecked(eval_len);
            (one, negative_one, positive_a, positive_b, inner)
        };
        ScratchLayout {
            one,
            negative_one,
            positive_a,
            positive_b,
            inner,
        }
    }
}
