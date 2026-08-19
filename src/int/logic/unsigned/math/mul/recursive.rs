//! Child-product dispatch shared by every Toom-Cook tier.
//!
//! A Toom evaluator produces two kinds of recursive product: a plain one, whose
//! operands are polynomial parts, and a *guarded* one, whose operands are
//! evaluations carrying a small high guard limb. Both live here once, taking the
//! [`TierCeiling`] and the guard bound as parameters, because per-tier copies
//! differ only in those two values and drift apart silently when one is retuned.
//!
//! The child dispatcher is taken as a parameter rather than derived from a
//! ceiling: Toom-3 hands its evaluations back to its own tier entry point,
//! while Toom-4 hands them to a ceiling-capped selection, and those are not the
//! same traversal.

use super::{AddMulKernel, ArchKernels, Limb, Multiplication, SharedEval, TierCeiling};

// ---------------------------------------------------------------------------
// Namespace
// ---------------------------------------------------------------------------

/// Namespace for child-product dispatch shared by every Toom-Cook tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Recursive;

impl Recursive {
    /// Multiply two polynomial parts with the tower capped at `ceiling`.
    ///
    /// `dst` may be wider than the exact product; the surplus is the caller's
    /// fixed-width guard and is cleared, because interpolation reads it.
    pub fn recursive_mul(
        dst: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        scratch: &mut [Limb],
        ceiling: TierCeiling,
    ) {
        if a.is_empty() || b.is_empty() {
            dst.fill(0);
            return;
        }
        let product_len = a
            .len()
            .checked_add(b.len())
            .expect("recursive product width overflowed");
        assert!(
            product_len <= dst.len(),
            "recursive product exceeds its fixed-width destination"
        );
        let (product, guard) = dst.split_at_mut(product_len);
        guard.fill(0);
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
    /// No guard fill here: `Multiplication::execute_square_plan` already clears whatever `dst`
    /// holds above the exact `2n`-limb square.
    pub fn recursive_sqr(dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb], ceiling: TierCeiling) {
        if a.is_empty() {
            dst.fill(0);
            return;
        }
        let square_len = a
            .len()
            .checked_mul(2)
            .expect("recursive square width overflowed");
        assert!(
            square_len <= dst.len(),
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
    /// number of radix chunks.
    ///
    /// `GUARD_BOUND` is the tier's proven bound on a single guard limb.
    /// `GUARD_LIMBS` is how many limbs the guard *product* occupies: one where
    /// `GUARD_BOUND^2` fits a limb (Toom-3 and Toom-4), two where the degree-six
    /// bound does not on every supported target (Toom-6.5 and Toom-8.5). The tier's
    /// destination must retain that many limbs above the exact low product.
    /// `multiply` is the tier's own child dispatcher.
    #[allow(
        unsafe_code,
        reason = "The const guard-width proof selects a prefix of a two-limb product"
    )]
    pub fn guarded_evaluation_product<const GUARD_BOUND: Limb, const GUARD_LIMBS: usize>(
        dst: &mut [Limb],
        evaluation_a: &[Limb],
        evaluation_b: &[Limb],
        scratch: &mut [Limb],
        kernel: AddMulKernel,
        multiply: impl FnOnce(&mut [Limb], &[Limb], &[Limb], &mut [Limb]),
    ) {
        const {
            assert!(
                GUARD_LIMBS == 1 || GUARD_LIMBS == 2,
                "a guard product spans one or two limbs"
            );
        }
        let Some((guard_a, low_a)) = evaluation_a.split_last() else {
            dst.fill(0);
            return;
        };
        let Some((guard_b, low_b)) = evaluation_b.split_last() else {
            dst.fill(0);
            return;
        };
        assert_eq!(low_a.len(), low_b.len(), "evaluation widths must match");
        assert!(
            *guard_a < GUARD_BOUND && *guard_b < GUARD_BOUND,
            "evaluation guard exceeds its proven bound"
        );
        let low_product_len = low_a
            .len()
            .checked_mul(2)
            .expect("guarded evaluation product width overflowed");
        let required_dst_len = low_product_len
            .checked_add(GUARD_LIMBS)
            .expect("guarded evaluation destination width overflowed");
        assert!(
            required_dst_len <= dst.len(),
            "guarded evaluation product exceeds its destination"
        );
        let (low_product, guard_space) = dst.split_at_mut(low_product_len);
        guard_space.fill(0);
        multiply(low_product, low_a, low_b, scratch);

        let (_, shifted_product) = dst.split_at_mut(low_a.len());
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low_b, *guard_a, kernel);
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low_a, *guard_b, kernel);
        let guard_product: [Limb; 2] = ArchKernels::mul_limb_lo_hi(*guard_a, *guard_b).into();
        assert!(
            GUARD_LIMBS == 2 || guard_product[1] == 0,
            "a one-limb guard product must not carry into a second limb"
        );
        // SAFETY: the const assertion proves `1 <= GUARD_LIMBS <= 2`, exactly
        // bounding this prefix of the two-limb scalar product.
        let guard_slice = unsafe { guard_product.get_unchecked(..GUARD_LIMBS) };
        // SAFETY: this routine's destination contract retains `GUARD_LIMBS`
        // limbs above `low_product_len`; `guard_slice` has exactly that width.
        let _ =
            unsafe { SharedEval::fused_add_shifted_in_place(dst, guard_slice, low_product_len) };
    }

    /// Square one guarded evaluation.
    ///
    /// The squaring specialization of [`Self::guarded_evaluation_product`]: the two cross
    /// terms coincide, so one scalar product at twice the guard replaces both.
    #[allow(
        unsafe_code,
        reason = "The const guard-width proof selects a prefix of a two-limb square"
    )]
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
        let Some((guard, low)) = evaluation.split_last() else {
            dst.fill(0);
            return;
        };
        assert!(
            *guard < GUARD_BOUND,
            "evaluation guard exceeds its proven bound"
        );
        let low_product_len = low
            .len()
            .checked_mul(2)
            .expect("guarded evaluation square width overflowed");
        let required_dst_len = low_product_len
            .checked_add(GUARD_LIMBS)
            .expect("guarded square destination width overflowed");
        assert!(
            required_dst_len <= dst.len(),
            "guarded evaluation square exceeds its destination"
        );
        let (low_square, guard_space) = dst.split_at_mut(low_product_len);
        guard_space.fill(0);
        square(low_square, low, scratch);

        let (_, shifted_product) = dst.split_at_mut(low.len());
        let doubled_guard = guard
            .checked_mul(2)
            .expect("guarded evaluation cross coefficient overflowed");
        SharedEval::add_mul_word_with_kernel_in_place(shifted_product, low, doubled_guard, kernel);
        let guard_square: [Limb; 2] = ArchKernels::mul_limb_lo_hi(*guard, *guard).into();
        assert!(
            GUARD_LIMBS == 2 || guard_square[1] == 0,
            "a one-limb guard square must not carry into a second limb"
        );
        // SAFETY: the const assertion proves `1 <= GUARD_LIMBS <= 2`, exactly
        // bounding this prefix of the two-limb scalar square.
        let guard_slice = unsafe { guard_square.get_unchecked(..GUARD_LIMBS) };
        // SAFETY: this routine's destination contract retains `GUARD_LIMBS`
        // limbs above `low_product_len`; `guard_slice` has exactly that width.
        let _ =
            unsafe { SharedEval::fused_add_shifted_in_place(dst, guard_slice, low_product_len) };
    }
}
