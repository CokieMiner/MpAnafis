//! Child-product dispatch shared by every Toom-Cook tier.
//!
//! Polynomial parts use a tier ceiling. Guarded evaluations split off their
//! bounded high limbs and multiply the low blocks recursively.
//!
//! The child dispatcher is taken as a parameter rather than derived from a
//! ceiling: Toom-3 hands its evaluations back to its own tier entry point,
//! while Toom-4 hands them to a ceiling-capped selection, and those are not the
//! same traversal.

#![expect(
    unsafe_code,
    reason = "Guarded recursive layouts retain complete initialized low blocks, bounded high guards, and disjoint reconstruction spans"
)]

use core::ptr::copy_nonoverlapping;

use super::{AddMulKernel, ArchKernels, Limb, LimbOutput, Multiplication, SharedEval, TierCeiling};

/// Namespace for child-product dispatch shared by every Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Recursive;

impl Recursive {
    /// Multiply two polynomial parts with the tower capped at `ceiling`.
    ///
    /// `dst` may be wider than the exact product; the surplus is the caller's
    /// fixed-width guard and is cleared, because interpolation reads it.
    pub fn recursive_mul(
        dst: &mut [impl LimbOutput],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        ceiling: TierCeiling,
    ) {
        if a.is_empty() || b.is_empty() {
            dst.fill(LimbOutput::from_limb(0));
            return;
        }
        // SAFETY: each slice spans at most isize::MAX bytes and each Limb
        // occupies at least two bytes on 16/32/64-bit targets. Thus the sum
        // of their limb counts is at most isize::MAX < usize::MAX.
        let product_len = unsafe { a.len().unchecked_add(b.len()) };
        debug_assert!(
            product_len <= dst.len(),
            "recursive product exceeds its fixed-width destination"
        );
        // SAFETY: every tier reserves the complete product_len-limb result
        // followed by its initialized interpolation guard suffix.
        let (product, guard) = unsafe { dst.split_at_mut_unchecked(product_len) };
        guard.fill(LimbOutput::from_limb(0));
        Multiplication::execute_plan(
            Multiplication::select_plan(a.len(), b.len(), ceiling),
            product,
            a,
            b,
            scratch,
        );
    }

    /// Square one polynomial part with the tower capped at `ceiling`.
    ///
    /// Square execution clears the surplus above the exact `2n`-limb result.
    pub fn recursive_sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb], ceiling: TierCeiling) {
        if a.is_empty() {
            dst.fill(0);
            return;
        }
        debug_assert!(
            // SAFETY: valid slice byte bounds and Limb >= 2 bytes prove
            // twice the input width fits usize on every supported target.
            unsafe { a.len().unchecked_mul(2) } <= dst.len(),
            "recursive square exceeds its fixed-width destination"
        );
        Multiplication::execute_square_plan(
            Multiplication::select_square_plan(a.len(), ceiling),
            dst,
            a,
            scratch,
        );
    }

    /// Multiply two guarded evaluations.
    ///
    /// For `x = low_x + guard_x*B^m` and `y = low_y + guard_y*B^m`,
    /// `x*y = low_x*low_y + (guard_x*low_y + guard_y*low_x)*B^m + guard_x*guard_y*B^2m`.
    /// Splitting the guard out keeps the recursive product at exactly `m` limbs per
    /// side instead of `m+1`, which is what makes the evaluation buffers a whole
    /// number of radix chunks. Both evaluation slices contain their complete
    /// low block and one initialized guard limb, even for a zero evaluation.
    ///
    /// `GUARD_BOUND` is the tier's proven bound on a single guard limb.
    /// `GUARD_LIMBS` is how many limbs the guard *product* occupies: one where
    /// `GUARD_BOUND^2` fits a limb (Toom-3 and Toom-4), two where the degree-six
    /// bound does not on every supported target (Toom-6.5 and Toom-8.5). The tier's
    /// destination must retain that many limbs above the exact low product.
    /// Both evaluations are below `GUARD_BOUND*B^m`, so the complete product
    /// and every nonnegative partial sum fit this guarded width. Seeding the
    /// high coefficient with the guard product avoids zeroing and adding it later.
    /// `multiply` is the tier's own child dispatcher.
    pub fn guarded_evaluation_product<
        const GUARD_BOUND: Limb,
        const GUARD_LIMBS: usize,
        Output: LimbOutput,
    >(
        dst: &mut [Output],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
        multiply: impl FnOnce(&mut [Output], &[Limb], &[Limb], &mut [Limb]),
    ) {
        const {
            assert!(
                GUARD_LIMBS == 1 || GUARD_LIMBS == 2,
                "a guard product spans one or two limbs"
            );
        }
        // SAFETY: each Toom layout supplies split_len+1 initialized evaluation
        // limbs, including the high guard even when the low block cancels to zero.
        let (guard_a, low_a) = unsafe { evaluation_a.split_last().unwrap_unchecked() };
        // SAFETY: the second operand has the same guarded evaluation layout.
        let (guard_b, low_b) = unsafe { evaluation_b.split_last().unwrap_unchecked() };
        debug_assert_eq!(low_a.len(), low_b.len(), "evaluation widths must match");
        debug_assert!(
            *guard_a < GUARD_BOUND && *guard_b < GUARD_BOUND,
            "evaluation guard exceeds its proven bound"
        );
        // SAFETY: evaluation_a is a valid nonempty limb slice and
        // low_a.len()+1 == evaluation_a.len(). The byte bound and Limb >= 2
        // bytes prove 2*low_a.len()+2 <= isize::MAX; GUARD_LIMBS <= 2.
        let low_product_len = unsafe { low_a.len().unchecked_mul(2) };
        debug_assert!(
            // SAFETY: the valid evaluation byte span proves 2*low_a.len()+2
            // fits usize, and the const assertion proves GUARD_LIMBS <= 2.
            unsafe { low_product_len.unchecked_add(GUARD_LIMBS) } <= dst.len(),
            "guarded evaluation product exceeds its destination"
        );
        let guard_product: [Limb; 2] = ArchKernels::mul_limb_lo_hi(*guard_a, *guard_b).into();
        debug_assert!(
            GUARD_LIMBS == 2 || guard_product[1] == 0,
            "a one-limb guard product must not carry into a second limb"
        );
        // SAFETY: the layout retains low_product_len+GUARD_LIMBS limbs and the const
        // assertion bounds GUARD_LIMBS by the two initialized scalar-product
        // limbs. The disjoint high prefix receives that exact coefficient;
        // only a surplus suffix is zero. The low recursive product cannot
        // overwrite the high coefficient, and later cross terms only add to it.
        let low_product = unsafe {
            let (low, guard_space) = dst.split_at_mut_unchecked(low_product_len);
            let (guard, surplus) = guard_space.split_at_mut_unchecked(GUARD_LIMBS);
            copy_nonoverlapping(
                guard_product.as_ptr(),
                guard.as_mut_ptr().cast(),
                GUARD_LIMBS,
            );
            surplus.fill(Output::from_limb(0));
            low
        };
        multiply(low_product, low_a, low_b, scratch);

        // SAFETY: multiply initialized every low-product limb; the guard copy
        // and surplus fill initialized the remaining disjoint suffix. Therefore
        // the complete destination is readable, and low_a.len() is in bounds.
        let (_, shifted_product) =
            unsafe { Output::assume_init_mut(dst).split_at_mut_unchecked(low_a.len()) };
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low_b, *guard_a, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low_a, *guard_b, kernel);
    }

    /// Square one guarded evaluation.
    ///
    /// The squaring specialization of [`Self::guarded_evaluation_product`]: the two cross
    /// terms coincide, so one scalar product at twice the guard replaces both.
    pub fn guarded_evaluation_square<const GUARD_BOUND: Limb, const GUARD_LIMBS: usize>(
        dst: &mut [Limb],
        evaluation: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
        square: impl FnOnce(&mut [Limb], &[Limb], &mut [Limb]),
    ) {
        const {
            assert!(
                GUARD_LIMBS == 1 || GUARD_LIMBS == 2,
                "a guard product spans one or two limbs"
            );
        }
        // SAFETY: each Toom square layout supplies split_len+1 initialized
        // evaluation limbs, including the guard for a zero low block.
        let (guard, low) = unsafe { evaluation.split_last().unwrap_unchecked() };
        debug_assert!(
            *guard < GUARD_BOUND,
            "evaluation guard exceeds its proven bound"
        );
        debug_assert!(
            *guard <= Limb::MAX.div_euclid(2),
            "doubled evaluation guard must fit one limb"
        );
        // SAFETY: low.len()+1 == evaluation.len(), whose valid byte span and
        // Limb >= 2 bytes prove 2*low.len()+2 <= isize::MAX. GUARD_LIMBS <= 2.
        let low_product_len = unsafe { low.len().unchecked_mul(2) };
        debug_assert!(
            // SAFETY: the valid evaluation byte span proves 2*low.len()+2
            // fits usize, and the const assertion proves GUARD_LIMBS <= 2.
            unsafe { low_product_len.unchecked_add(GUARD_LIMBS) } <= dst.len(),
            "guarded evaluation square exceeds its destination"
        );
        let guard_square: [Limb; 2] = ArchKernels::mul_limb_lo_hi(*guard, *guard).into();
        debug_assert!(
            GUARD_LIMBS == 2 || guard_square[1] == 0,
            "a one-limb guard square must not carry into a second limb"
        );
        // SAFETY: the layout reserves the full low square and GUARD_LIMBS high
        // limbs; the const assertion bounds the copy by the initialized scalar
        // square. Disjoint partitions keep this seeded high coefficient outside
        // the low recursive square. Surplus interpolation guards need zeros.
        let low_square = unsafe {
            let (low_square, guard_space) = dst.split_at_mut_unchecked(low_product_len);
            let (high, surplus) = guard_space.split_at_mut_unchecked(GUARD_LIMBS);
            copy_nonoverlapping(guard_square.as_ptr(), high.as_mut_ptr(), GUARD_LIMBS);
            surplus.fill(0);
            low_square
        };
        square(low_square, low, scratch);

        // SAFETY: low.len() <= low_product_len <= dst.len().
        let (_, shifted_product) = unsafe { dst.split_at_mut_unchecked(low.len()) };
        // SAFETY: Toom-3/4/6 guards are below 7/15/5462, all below MAX/2
        // even on 16-bit targets. Toom-8 uses this one-guard square only on
        // 32/64-bit targets, where its evaluation guard is below 2^25 < MAX/2;
        // its 16-bit two-guard evaluations use the full recursive square.
        let doubled_guard = unsafe { guard.unchecked_mul(2) };
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low, doubled_guard, kernel);
    }
}
