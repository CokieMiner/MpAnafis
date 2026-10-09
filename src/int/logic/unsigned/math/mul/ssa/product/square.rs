//! Pointwise transform squaring and coefficient traversal.

#![expect(
    unsafe_code,
    reason = "Complete coefficient spans and admitted leaf arenas bound in-place squares and disjoint recursive worker partitions"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{
    Limb, PointwiseSquarePlan, PointwiseSquareStrategy, Residue, SsaPlan, SsaPointwise, SsaRing,
    SsaTransform,
};

impl SsaPointwise {
    /// Squares every coefficient of a transformed matrix in place.
    ///
    /// # Safety
    /// `matrix` holds `transform_len` complete `SsaRing::coeff_limbs(mod_bits)`-limb
    /// coefficients. `leaf_budget` is the enclosing transform's admitted power
    /// of two. `sqr_scratch` covers that budget after reduction to the active
    /// prefix's largest power of two.
    pub unsafe fn pointwise_square_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        leaf_budget: NonZeroUsize,
        plan: &PointwiseSquarePlan,
        executor: &E,
        sqr_scratch: &mut [Limb],
    ) {
        let mod_bits = plan.bits;
        let cl = SsaRing::coeff_limbs(mod_bits).get();
        if leaf_budget.get() > 1
            && transform_len >= 2
            && SsaTransform::has_parallel_work(
                transform_len,
                SsaPlan::basecase_product_cost(cl),
                executor.parallelism().get(),
            )
        {
            let needed_scratch = plan.scratch_len.get();
            // SAFETY: transform_len>=2 bounds leading_zeros below
            // usize::BITS-1; the floor-power exponent and shift fit.
            let active_leaves = unsafe {
                let log = usize::BITS
                    .unchecked_sub(1)
                    .unchecked_sub(transform_len.leading_zeros());
                1_usize.unchecked_shl(log)
            };
            let leaf_count = leaf_budget.get().min(active_leaves);
            // SAFETY: leaf_count<=leaf_budget, whose complete square workspace
            // product was checked at the enclosing transform admission.
            let required_scratch = unsafe { needed_scratch.unchecked_mul(leaf_count) };
            debug_assert!(
                sqr_scratch.len() >= required_scratch,
                "pointwise square scratch must be partitioned at the outer transform boundary"
            );
            // SAFETY: matrix and scratch are complete, coefficient-aligned partitions.
            unsafe {
                pointwise_square_parallel(
                    matrix,
                    transform_len,
                    mod_bits,
                    &plan.strategy,
                    needed_scratch,
                    leaf_count,
                    sqr_scratch,
                    executor,
                );
            }
            return;
        }

        // SAFETY: the matrix contains transform_len complete coefficients;
        // sqr_scratch covers the selected sequential square strategy.
        unsafe {
            pointwise_square_sequential(
                matrix,
                transform_len,
                mod_bits,
                &plan.strategy,
                sqr_scratch,
            );
        }
    }

    /// The squaring counterpart of special residue handling: `0^2 = 0` and `(-1)^2 = 1`.
    ///
    /// # Safety
    /// `dst` spans at least `cl` limbs and `cl` is non-zero.
    pub unsafe fn write_special_residue_square(
        dst: &mut [Limb],
        cl: usize,
        class: Residue,
    ) -> bool {
        if class == Residue::Ordinary {
            return false;
        }
        // SAFETY: cl>0 and dst spans cl limbs. Every special output writes its
        // low digit once and establishes the remaining zero data and guard.
        unsafe {
            *dst.get_unchecked_mut(0) = Limb::from(class == Residue::NegOne);
            dst.get_unchecked_mut(1..cl).fill(0);
        }
        true
    }
}

/// Executes pointwise squaring in disjoint coefficient ranges.
///
/// # Safety
/// The matrix contains `transform_len` complete coefficients. Recursive
/// splits occur only on coefficient boundaries; every leaf owns an independent
/// caller-provided scratch range before calling the shared sequential kernel.
#[expect(
    clippy::too_many_arguments,
    reason = "The recursive worker carries one immutable plan and one executor alongside the disjoint matrix ranges"
)]
unsafe fn pointwise_square_parallel<E: ParallelExecutor>(
    matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    plan: &PointwiseSquareStrategy,
    needed_scratch: usize,
    leaf_count: usize,
    scratch: &mut [Limb],
    executor: &E,
) {
    let cl = SsaRing::coeff_limbs(mod_bits).get();
    if leaf_count == 1 {
        debug_assert!(
            scratch.len() >= needed_scratch,
            "validated pointwise square leaf scratch must cover its workspace"
        );
        // SAFETY: the outer preparation proved one complete arena per leaf.
        let leaf_scratch = unsafe { scratch.get_unchecked_mut(..needed_scratch) };
        // SAFETY: this leaf is a complete coefficient-aligned matrix partition.
        unsafe {
            pointwise_square_sequential(matrix, transform_len, mod_bits, plan, leaf_scratch);
        }
        return;
    }

    let left_count = transform_len.div_euclid(2);
    let left_leaves = leaf_count >> 1;
    // SAFETY: matrix contains transform_len*cl limbs, and the checked arena
    // contains leaf_count workspaces. Both half partitions are bounded by them.
    let (right_count, (left_matrix, right_matrix), (left_scratch, right_scratch)) = unsafe {
        let left_limbs = left_count.unchecked_mul(cl);
        let left_scratch_len = needed_scratch.unchecked_mul(left_leaves);
        (
            transform_len.unchecked_sub(left_count),
            matrix.split_at_mut_unchecked(left_limbs),
            scratch.split_at_mut_unchecked(left_scratch_len),
        )
    };
    let ((), ()) = executor.join(
        || {
            // SAFETY: the first matrix range is disjoint and complete.
            unsafe {
                pointwise_square_parallel(
                    left_matrix,
                    left_count,
                    mod_bits,
                    plan,
                    needed_scratch,
                    left_leaves,
                    left_scratch,
                    executor,
                );
            }
        },
        || {
            // SAFETY: the second matrix range is disjoint and complete.
            unsafe {
                pointwise_square_parallel(
                    right_matrix,
                    right_count,
                    mod_bits,
                    plan,
                    needed_scratch,
                    left_leaves,
                    right_scratch,
                    executor,
                );
            }
        },
    );
}

/// Sequential loop for pointwise squaring over a contiguous chunk of coefficients.
///
/// # Safety
/// `matrix` holds `transform_len` complete `SsaRing::coeff_limbs(mod_bits)`-limb
/// coefficients and `sqr_scratch` covers the selected strategy's arena.
unsafe fn pointwise_square_sequential(
    matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    strategy: &PointwiseSquareStrategy,
    sqr_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits).get();
    // SAFETY: the admitted positive ring has at least one data limb and its
    // guard, so cl>=2 and cl-1 is the guard's initialized index.
    let guard_index = unsafe { cl.unchecked_sub(1) };
    match strategy {
        PointwiseSquareStrategy::Basecase(plan) => {
            for index in 0..transform_len {
                // SAFETY: index is inside the complete coefficient matrix.
                let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl) };
                // SAFETY: slot and scratch are complete and disjoint. After
                // normalization, guard one denotes B^ml with all data zero;
                // its square needs only the low digit and guard overwritten.
                unsafe {
                    if SsaRing::normalize(slot, mod_bits) {
                        *slot.get_unchecked_mut(0) = 1;
                        *slot.get_unchecked_mut(guard_index) = 0;
                        continue;
                    }
                    // Normalization returned an ordinary residue, whose guard
                    // is zero. A zero data prefix already contains its square.
                    if slot
                        .get_unchecked(..guard_index)
                        .iter()
                        .all(|&limb| limb == 0)
                    {
                        continue;
                    }
                    SsaPointwise::fermat_basecase_sqr_assign(slot, mod_bits, *plan, sqr_scratch);
                }
            }
        }
        PointwiseSquareStrategy::Transform(plan) => {
            for index in 0..transform_len {
                // SAFETY: index is inside the complete coefficient matrix.
                let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl) };
                // SAFETY: slot and scratch are disjoint, and the retained plan
                // matches this modulus and its complete sequential arena.
                // Normalization makes guard one imply zero data; its in-place
                // square overwrites only the low digit and guard.
                unsafe {
                    if SsaRing::normalize(slot, mod_bits) {
                        *slot.get_unchecked_mut(0) = 1;
                        *slot.get_unchecked_mut(guard_index) = 0;
                        continue;
                    }
                    // The ordinary zero residue remains zero in place; neither
                    // the recursive plan nor its arena needs to be touched.
                    if slot
                        .get_unchecked(..guard_index)
                        .iter()
                        .all(|&limb| limb == 0)
                    {
                        continue;
                    }
                    plan.sqr_assign(slot, sqr_scratch);
                }
            }
        }
    }
}
