//! The Toom-Cook 3 driver: split, evaluate, recurse, interpolate.
//!
//! # References
//!
//! - Bodrato, M. (2007). *Towards Optimal Toom-Cook Multiplication for
//!   Univariate and Multivariate Polynomials in Characteristic 2 and 0*.
//!   WAIFI 2007, LNCS 4547, 116–133.
//!   <https://doi.org/10.1007/978-3-540-73074-3_10>.

#![expect(
    unsafe_code,
    reason = "Validated three-part layouts reserve disjoint guarded evaluations, endpoint products, and recursive scratch"
)]

use core::cmp::max;

use super::{
    AddMulKernel, ArchKernels, Karatsuba, Limb, LimbOutput, MiddleValues, Multiplication,
    Recursive, SQR_TOOM_COOK_THRESHOLD, TOOM_COOK_THRESHOLD,
};

/// Namespace for the three-way Toom-Cook multiplication and squaring tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Toom3;

struct ScratchLayout<'buffer> {
    one: &'buffer mut [Limb],
    neg_one: &'buffer mut [Limb],
    two: &'buffer mut [Limb],
    eval_a: &'buffer mut [Limb],
    eval_b: &'buffer mut [Limb],
    inner: &'buffer mut [Limb],
}

impl Toom3 {
    /// Selects Toom-3 or the lower tower at the configured product crossover.
    ///
    /// `dst` must hold the full product and `scratch` must have at least
    /// [`Multiplication::toom3_dispatch_mul_scratch_len`] limbs.
    pub fn dispatch_mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < TOOM_COOK_THRESHOLD
            || b.len() < TOOM_COOK_THRESHOLD
            || a.len() < 3
            || b.len() < 3
        {
            Karatsuba::dispatch_mul(dst, a, b, scratch);
            return;
        }
        Self::mul(dst, a, b, scratch);
    }

    /// Executes one Toom-Cook 3 level regardless of the configured crossover.
    ///
    /// Recursive products retain normal dispatch. Inputs shorter than three
    /// limbs use the lower tower because a three-way split is impossible.
    pub fn mul(dst: &mut [impl LimbOutput], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 3 || b.len() < 3 {
            Karatsuba::dispatch_mul(dst, a, b, scratch);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::toom3_mul_scratch_len(a.len(), b.len()),
            "Toom-3 multiplication scratch is undersized: have {}, need {} for {}x{} limbs",
            scratch.len(),
            Multiplication::toom3_mul_scratch_len(a.len(), b.len()),
            a.len(),
            b.len()
        );
        let split_len = max(a.len(), b.len()).div_ceil(3);
        let (a0, a1, a2) = Self::split_three(a, split_len);
        let (b0, b1, b2) = Self::split_three(b, split_len);
        // SAFETY: m=ceil(max(|a|,|b|)/3) for actual limb slices. Their byte
        // lengths are <=isize::MAX and limbs occupy >=2 bytes; 4m and 2m+1
        // therefore fit usize on 16-, 32-, and 64-bit targets.
        let (product_len, eval_len, high_offset) = unsafe {
            (
                split_len.unchecked_mul(2).unchecked_add(1),
                split_len.unchecked_add(1),
                split_len.unchecked_mul(4),
            )
        };

        let ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            eval_b,
            inner,
        } = Self::split_scratch::<false>(scratch, product_len, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        // Negative evaluations occupy product slots that are dead until the
        // negative-point product consumes them. Positive evaluations stay in
        // the reusable operand buffers through both the +1 and +2 products.
        // SAFETY: each product slot has 2m+1 limbs, containing its m+1-limb
        // temporary evaluation. Both temporaries die before the slots are reused.
        let (negative_eval_a, negative_eval_b) = unsafe {
            (
                one.split_at_mut_unchecked(eval_len).0,
                two.split_at_mut_unchecked(eval_len).0,
            )
        };
        let negative_a = Self::evaluate_one_and_negative_one(eval_a, negative_eval_a, a0, a1, a2);
        let negative_b = Self::evaluate_one_and_negative_one(eval_b, negative_eval_b, b0, b1, b2);
        Self::mul_evaluation(
            neg_one,
            negative_eval_a,
            negative_eval_b,
            inner,
            add_mul_kernel,
        );

        Self::mul_evaluation(one, eval_a, eval_b, inner, add_mul_kernel);

        Self::evaluate_two_from_one(eval_a, a1, a2, add_mul_kernel);
        Self::evaluate_two_from_one(eval_b, b1, b2, add_mul_kernel);
        Self::mul_evaluation(two, eval_a, eval_b, inner, add_mul_kernel);

        // SAFETY: the low endpoint width is at most the validated |a|+|b|
        // output width because each constant is a prefix of its operand.
        let low_product_len = unsafe { a0.len().unchecked_add(b0.len()) };
        let high_product_len = if a2.is_empty() || b2.is_empty() {
            0
        } else {
            // SAFETY: each high part is a suffix of its operand, so their
            // combined width is bounded by the validated complete product.
            unsafe { a2.len().unchecked_add(b2.len()) }
        };
        {
            // SAFETY: the full product buffer contains the low endpoint.
            let (zero, _) = unsafe { dst.split_at_mut_unchecked(low_product_len) };
            Self::dispatch_mul(zero, a0, b0, inner);
        }
        if high_product_len != 0 {
            // SAFETY: both high parts are nonempty here, so the infinity
            // endpoint occupies [4m,|a|+|b|), within the validated output.
            let (infinity, _) = unsafe {
                dst.split_at_mut_unchecked(high_offset)
                    .1
                    .split_at_mut_unchecked(high_product_len)
            };
            Self::dispatch_mul(infinity, a2, b2, inner);
        }
        Self::clear_destination_outside_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );
        // SAFETY: the recursive endpoints initialized their exact disjoint
        // spans; clearing the intervening and trailing gaps wrote every other
        // destination element. Interpolation therefore receives readable limbs.
        let initialized = unsafe { LimbOutput::assume_init_mut(dst) };

        Self::interpolate_endpoints(
            initialized,
            low_product_len,
            high_offset,
            high_product_len,
            MiddleValues {
                one: &mut *one,
                neg_one: &mut *neg_one,
                two: &mut *two,
                neg_one_negative: negative_a ^ negative_b,
            },
        );
        Self::reconstruct_middle(initialized, split_len, neg_one, one, two);
    }

    /// Selects Toom-3 or the lower tower at the configured square crossover.
    pub fn dispatch_sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < SQR_TOOM_COOK_THRESHOLD || a.len() < 3 {
            Karatsuba::dispatch_sqr(dst, a, scratch);
            return;
        }
        Self::sqr(dst, a, scratch);
    }

    /// Executes one Toom-Cook 3 square level regardless of the crossover.
    ///
    /// Recursive squares still use normal dispatch. Inputs shorter than three
    /// limbs retain the Karatsuba/basecase fallback because a three-way split is
    /// impossible.
    pub fn sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) {
        if a.len() < 3 {
            Karatsuba::dispatch_sqr(dst, a, scratch);
            return;
        }
        debug_assert!(
            scratch.len() >= Multiplication::toom3_sqr_scratch_len(a.len()),
            "Toom-3 squaring scratch is undersized: have {}, need {} for {} limbs",
            scratch.len(),
            Multiplication::toom3_sqr_scratch_len(a.len()),
            a.len()
        );
        let split_len = a.len().div_ceil(3);
        let (a0, a1, a2) = Self::split_three(a, split_len);
        // SAFETY: m=ceil(|a|/3), |a|*size_of::<Limb>()<=isize::MAX and
        // size_of::<Limb>()>=2. Thus 4m, 2m+1, and m+1 fit every supported usize.
        let (product_len, eval_len, high_offset) = unsafe {
            (
                split_len.unchecked_mul(2).unchecked_add(1),
                split_len.unchecked_add(1),
                split_len.unchecked_mul(4),
            )
        };

        let ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            inner,
            ..
        } = Self::split_scratch::<true>(scratch, product_len, eval_len);
        let add_mul_kernel = ArchKernels::selected_add_mul_limbs_unchecked();

        // SAFETY: the 2m+1-limb product slot contains the m+1-limb evaluation.
        let (negative_eval, _) = unsafe { one.split_at_mut_unchecked(eval_len) };
        let _ = Self::evaluate_one_and_negative_one(eval_a, negative_eval, a0, a1, a2);
        Self::sqr_evaluation(neg_one, negative_eval, inner, add_mul_kernel);

        Self::sqr_evaluation(one, eval_a, inner, add_mul_kernel);
        Self::evaluate_two_from_one(eval_a, a1, a2, add_mul_kernel);
        Self::sqr_evaluation(two, eval_a, inner, add_mul_kernel);

        // SAFETY: both doubled endpoint lengths fit the validated 2|a|-limb
        // square destination; doubling a real limb-slice length also fits usize.
        let (low_product_len, high_product_len) =
            unsafe { (a0.len().unchecked_mul(2), a2.len().unchecked_mul(2)) };
        {
            // SAFETY: the low endpoint occupies a prefix of the full square.
            let (zero, _) = unsafe { dst.split_at_mut_unchecked(low_product_len) };
            Self::dispatch_sqr(zero, a0, inner);
        }
        if high_product_len != 0 {
            // SAFETY: a nonempty a2 makes [4m,2|a|) the exact infinity square
            // range, contained in the validated destination.
            let (infinity, _) = unsafe {
                dst.split_at_mut_unchecked(high_offset)
                    .1
                    .split_at_mut_unchecked(high_product_len)
            };
            Self::dispatch_sqr(infinity, a2, inner);
        }
        Self::clear_destination_outside_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
        );

        Self::interpolate_endpoints(
            dst,
            low_product_len,
            high_offset,
            high_product_len,
            MiddleValues {
                one: &mut *one,
                neg_one: &mut *neg_one,
                two: &mut *two,
                neg_one_negative: false,
            },
        );
        Self::reconstruct_middle(dst, split_len, neg_one, one, two);
    }

    /// Toom-3 evaluations carry a guard below seven; recursion stays in this tier.
    fn mul_evaluation(
        dst: &mut [Limb],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_product::<7, 1, _>(
            dst,
            evaluation_a,
            evaluation_b,
            scratch,
            add_mul_kernel,
            Self::dispatch_mul,
        );
    }

    fn sqr_evaluation(
        dst: &mut [Limb],
        evaluation: &[Limb],
        scratch: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        Recursive::guarded_evaluation_square::<7, 1>(
            dst,
            evaluation,
            scratch,
            add_mul_kernel,
            Self::dispatch_sqr,
        );
    }

    fn clear_destination_outside_endpoints(
        dst: &mut [impl LimbOutput],
        low_product_len: usize,
        high_offset: usize,
        high_product_len: usize,
    ) {
        if high_product_len == 0 {
            // SAFETY: the low endpoint is a prefix of the full product.
            let (_, unwritten) = unsafe { dst.split_at_mut_unchecked(low_product_len) };
            unwritten.fill(LimbOutput::from_limb(0));
            return;
        }

        // Both recursive endpoint products overwrite their complete ranges. Only
        // the gap between them and the tail above the exact infinity product must
        // start at zero before the overlapping coefficients are reconstructed.
        // SAFETY: a nonzero high product means both high operand blocks exist.
        // Thus low_product_len<=2m<=4m and 4m+high_product_len<=dst.len().
        let (middle_gap, trailing_gap) = unsafe {
            let (before_high, high_and_after) = dst.split_at_mut_unchecked(high_offset);
            (
                before_high.split_at_mut_unchecked(low_product_len).1,
                high_and_after.split_at_mut_unchecked(high_product_len).1,
            )
        };
        middle_gap.fill(LimbOutput::from_limb(0));
        trailing_gap.fill(LimbOutput::from_limb(0));
    }

    /// Partitions three point products, operand evaluations, and child scratch.
    const fn split_scratch<const SQUARE: bool>(
        scratch: &mut [Limb],
        product_len: usize,
        eval_len: usize,
    ) -> ScratchLayout<'_> {
        let second_eval_len = if SQUARE { 0 } else { eval_len };
        // SAFETY: dispatch sizing reserves three (2m+1)-limb
        // products and one/two (m+1)-limb evaluations, followed by child scratch.
        // Sequential splits retain initialization and make all spans disjoint.
        let (one, neg_one, two, eval_a, eval_b, inner) = unsafe {
            let (one, after_one) = scratch.split_at_mut_unchecked(product_len);
            let (neg_one, after_neg_one) = after_one.split_at_mut_unchecked(product_len);
            let (two, after_two) = after_neg_one.split_at_mut_unchecked(product_len);
            let (eval_a, after_eval_a) = after_two.split_at_mut_unchecked(eval_len);
            let (eval_b, inner) = after_eval_a.split_at_mut_unchecked(second_eval_len);
            (one, neg_one, two, eval_a, eval_b, inner)
        };
        ScratchLayout {
            one,
            neg_one,
            two,
            eval_a,
            eval_b,
            inner,
        }
    }

    const fn split_three(values: &[Limb], split_len: usize) -> (&[Limb], &[Limb], &[Limb]) {
        // SAFETY: m=ceil(maximum actual operand length/3), so 2m fits usize
        // from the isize::MAX byte bound and at least two bytes per limb.
        let two_parts = unsafe { split_len.unchecked_mul(2) };
        if values.len() > two_parts {
            // SAFETY: the branch establishes values.len()>2m.
            let (part0, part1, part2) = unsafe {
                let (part0, after_part0) = values.split_at_unchecked(split_len);
                let (part1, part2) = after_part0.split_at_unchecked(split_len);
                (part0, part1, part2)
            };
            (part0, part1, part2)
        } else if values.len() > split_len {
            // SAFETY: this branch establishes values.len()>m.
            let (part0, part1) = unsafe { values.split_at_unchecked(split_len) };
            (part0, part1, &[])
        } else {
            (values, &[], &[])
        }
    }
}
