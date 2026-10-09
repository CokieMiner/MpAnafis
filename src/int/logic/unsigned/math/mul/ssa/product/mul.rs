//! Pointwise transform multiplication and coefficient traversal.

#![expect(
    unsafe_code,
    reason = "Admitted coefficient counts and leaf budgets bound disjoint matrix partitions, product arenas, and initialized special-residue writes"
)]

use core::{num::NonZeroUsize, ptr::copy_nonoverlapping};

use crate::parallel::ParallelExecutor;

use super::{
    Limb, LimbOutput, PointwiseMulPlan, PointwiseMulStrategy, Residue, SsaPlan, SsaPointwise,
    SsaRing, SsaTransform,
};

impl SsaPointwise {
    /// Multiplies every coefficient pair using the supplied synchronous executor.
    ///
    /// Both sequential and parallel paths use the caller-provided scratch.
    /// Parallel leaves receive disjoint coefficient-aligned regions; nested
    /// coefficient products use a sequential child executor to avoid recursive
    /// oversubscription.
    ///
    /// # Safety
    /// The matrices hold `transform_len` complete coefficients. `leaf_budget`
    /// is the enclosing transform's admitted power of two. Scratch covers that
    /// budget after reduction to the active prefix's largest power of two.
    pub unsafe fn pointwise_multiply_with_executor<E: ParallelExecutor>(
        left_matrix: &mut [Limb],
        right_matrix: &mut [Limb],
        transform_len: usize,
        leaf_budget: NonZeroUsize,
        plan: &PointwiseMulPlan,
        executor: &E,
        product_scratch: &mut [Limb],
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
            // SAFETY: transform_len>=2 gives leading_zeros<=usize::BITS-2,
            // so the floor-power exponent and its shift are representable.
            let active_leaves = unsafe {
                let log = usize::BITS
                    .unchecked_sub(1)
                    .unchecked_sub(transform_len.leading_zeros());
                1_usize.unchecked_shl(log)
            };
            let leaf_count = leaf_budget.get().min(active_leaves);
            // SAFETY: leaf_count<=leaf_budget, whose complete workspace product
            // was checked at the enclosing transform admission.
            let required_scratch = unsafe { needed_scratch.unchecked_mul(leaf_count) };
            debug_assert!(
                product_scratch.len() >= required_scratch,
                "pointwise scratch must be partitioned at the outer transform boundary"
            );
            // SAFETY: the matrices and caller-owned scratch contain complete,
            // coefficient-aligned partitions for every leaf.
            unsafe {
                pointwise_multiply_parallel(
                    left_matrix,
                    right_matrix,
                    transform_len,
                    mod_bits,
                    &plan.strategy,
                    needed_scratch,
                    leaf_count,
                    product_scratch,
                    executor,
                );
            }
            return;
        }

        // SAFETY: both disjoint matrices contain transform_len complete
        // coefficients; the caller supplies the selected sequential arena.
        unsafe {
            pointwise_multiply_sequential(
                left_matrix,
                right_matrix,
                transform_len,
                mod_bits,
                &plan.strategy,
                product_scratch,
            );
        }
    }

    /// Writes the product when either operand is a special residue, and reports
    /// whether it did.
    ///
    /// Zero absorbs and `-1` negates, so both cases are a fill or a negated copy.
    /// Neither needs a transform, and both are reachable at every level of the
    /// recursion, so this guard sits in front of the basecase as well as in front
    /// of the transform.
    ///
    /// # Safety
    /// `dst`, `left`, and `right` each span at least `SsaRing::mod_limbs(mod_bits) + 1`
    /// limbs, and `dst` is disjoint from both operands.
    pub unsafe fn write_special_residue_product(
        dst: &mut [impl LimbOutput],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
    ) -> bool {
        let ml = SsaRing::mod_limbs(mod_bits);
        // SAFETY: ml=mod_bits/LIMB_BITS and LIMB_BITS>=16, so ml+1 fits usize.
        let cl = unsafe { ml.unchecked_add(1) };
        // SAFETY: ml < cl and the caller guarantees both spans have cl limbs.
        let left_class = unsafe { SsaRing::classify_residue(left, ml) };
        // SAFETY: same bounds proof as left.
        let right_class = unsafe { SsaRing::classify_residue(right, ml) };

        if left_class == Residue::Zero || right_class == Residue::Zero {
            // SAFETY: the caller guarantees dst spans cl writable limbs.
            unsafe { dst.get_unchecked_mut(..cl) }.fill(LimbOutput::from_limb(0));
            return true;
        }
        if left_class != Residue::NegOne && right_class != Residue::NegOne {
            return false;
        }
        // Multiplying by -1 is a negated copy of the other operand.
        let source_coefficient = if left_class == Residue::NegOne {
            right
        } else {
            left
        };
        // SAFETY: `dst` contains the complete writable coefficient span.
        let destination = unsafe { dst.get_unchecked_mut(..cl) };
        // SAFETY: `source` contains the complete initialized coefficient span.
        let source = unsafe { source_coefficient.get_unchecked(..cl) };
        // SAFETY: the disjoint complete spans have identical widths. LimbOutput
        // preserves native limb layout, so this initializes the whole coefficient.
        unsafe {
            copy_nonoverlapping(source.as_ptr(), destination.as_mut_ptr().cast(), cl);
        }
        // SAFETY: the copy initialized every coefficient limb; this exclusive
        // borrow covers exactly cl elements and mod_bits matches that width.
        unsafe {
            SsaRing::negate(LimbOutput::assume_init_mut(destination), mod_bits);
        }
        true
    }

    /// Handles the sole pointwise representation outside ordinary fixed-width products.
    ///
    /// Canonical zero has a zero guard; every tier writes its zero product.
    /// Canonical `-1` alone has guard one, so no data-limb scan is required.
    ///
    /// # Safety
    /// `left` and `right` are disjoint, canonical `SsaRing::coeff_limbs(mod_bits)`
    /// spans. The flags are their `normalize` results, retained before any write.
    pub unsafe fn write_pointwise_special_product(
        left: &mut [Limb],
        right: &[Limb],
        mod_bits: usize,
        left_neg_one: bool,
        right_neg_one: bool,
    ) -> bool {
        let ml = SsaRing::mod_limbs(mod_bits);
        if left_neg_one {
            // SAFETY: ml=mod_bits/LIMB_BITS and LIMB_BITS>=16, so ml+1 fits usize.
            let cl = unsafe { ml.unchecked_add(1) };
            // SAFETY: `left` contains the complete writable coefficient span.
            let destination = unsafe { left.get_unchecked_mut(..cl) };
            // SAFETY: `right` is a disjoint complete initialized coefficient span.
            let source = unsafe { right.get_unchecked(..cl) };
            destination.copy_from_slice(source);
            // SAFETY: left now contains right's canonical residue.
            unsafe {
                SsaRing::negate(left, mod_bits);
            }
            return true;
        }
        if right_neg_one {
            // SAFETY: left is canonical, so in-place negation is valid.
            unsafe {
                SsaRing::negate(left, mod_bits);
            }
            return true;
        }
        false
    }
}

/// Executes pointwise products in disjoint coefficient ranges.
///
/// # Safety
/// The two matrices contain `transform_len` complete coefficients. Recursive
/// splits occur only on coefficient boundaries; every leaf owns an independent
/// caller-provided scratch range before calling the shared sequential kernel.
#[expect(
    clippy::too_many_arguments,
    reason = "The recursive worker carries one immutable plan and one executor alongside the two disjoint matrix ranges"
)]
unsafe fn pointwise_multiply_parallel<E: ParallelExecutor>(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    plan: &PointwiseMulStrategy,
    needed_scratch: usize,
    leaf_count: usize,
    scratch: &mut [Limb],
    executor: &E,
) {
    let cl = SsaRing::coeff_limbs(mod_bits).get();
    if leaf_count == 1 {
        debug_assert!(
            scratch.len() >= needed_scratch,
            "validated pointwise leaf scratch must cover its product workspace"
        );
        // SAFETY: the outer preparation proved one complete arena per leaf.
        let leaf_scratch = unsafe { scratch.get_unchecked_mut(..needed_scratch) };
        // SAFETY: this leaf is a complete coefficient-aligned matrix partition.
        unsafe {
            pointwise_multiply_sequential(
                left_matrix,
                right_matrix,
                transform_len,
                mod_bits,
                plan,
                leaf_scratch,
            );
        }
        return;
    }

    let left_count = transform_len.div_euclid(2);
    let left_leaves = leaf_count >> 1;
    // SAFETY: the complete matrices contain transform_len*cl limbs, and the
    // checked caller arena contains leaf_count workspaces. Halves stay within
    // those bounds, including odd truncated frequency counts.
    let (
        right_count,
        (left_first, left_second),
        (right_first, right_second),
        (left_scratch, right_scratch),
    ) = unsafe {
        let left_limbs = left_count.unchecked_mul(cl);
        let left_scratch_len = needed_scratch.unchecked_mul(left_leaves);
        (
            transform_len.unchecked_sub(left_count),
            left_matrix.split_at_mut_unchecked(left_limbs),
            right_matrix.split_at_mut_unchecked(left_limbs),
            scratch.split_at_mut_unchecked(left_scratch_len),
        )
    };
    let ((), ()) = executor.join(
        || {
            // SAFETY: the first matrix ranges are disjoint and complete.
            unsafe {
                pointwise_multiply_parallel(
                    left_first,
                    right_first,
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
            // SAFETY: the second matrix ranges are disjoint and complete.
            unsafe {
                pointwise_multiply_parallel(
                    left_second,
                    right_second,
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

/// Sequential loop for pointwise multiplication over a contiguous chunk of coefficients.
///
/// # Safety
/// - `left_matrix` and `right_matrix` each have `transform_len * SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `product_scratch` covers the selected strategy's complete arena; only
///   negacyclic products require a staged coefficient.
unsafe fn pointwise_multiply_sequential(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    strategy: &PointwiseMulStrategy,
    product_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits).get();
    match strategy {
        PointwiseMulStrategy::Negacyclic(plan) => {
            // SAFETY: the selected negacyclic strategy reserves one complete
            // result coefficient before its product arena.
            let (result, mul_scratch) = unsafe { product_scratch.split_at_mut_unchecked(cl) };
            for index in 0..transform_len {
                // SAFETY: offset + cl <= matrix length by construction; both matrices
                // are disjoint and each holds transform_len complete coefficients.
                let (left, right) = unsafe {
                    let offset = index.unchecked_mul(cl);
                    let end = offset.unchecked_add(cl);
                    (
                        left_matrix.get_unchecked_mut(offset..end),
                        right_matrix.get_unchecked_mut(offset..end),
                    )
                };

                // SAFETY: both spans are complete cl-limb coefficients.
                unsafe {
                    let left_neg_one = SsaRing::normalize(left, mod_bits);
                    let right_neg_one = SsaRing::normalize(right, mod_bits);
                    if SsaPointwise::write_pointwise_special_product(
                        left,
                        right,
                        mod_bits,
                        left_neg_one,
                        right_neg_one,
                    ) {
                        continue;
                    }
                    plan.mul_assign_left(left, right, result, mul_scratch);
                }
            }
        }
        PointwiseMulStrategy::Basecase(plan) => {
            for index in 0..transform_len {
                // SAFETY: index is within bounds of transform_len complete coefficients.
                let left = unsafe { SsaTransform::coeff_mut(left_matrix, index, cl) };
                // SAFETY: same bounds proof as left, applied to the right matrix.
                let right = unsafe { SsaTransform::coeff_mut(right_matrix, index, cl) };

                // SAFETY: both coefficients are complete and disjoint; the scratch
                // meets the plan requirement.
                unsafe {
                    let left_neg_one = SsaRing::normalize(left, mod_bits);
                    let right_neg_one = SsaRing::normalize(right, mod_bits);
                    if SsaPointwise::write_pointwise_special_product(
                        left,
                        right,
                        mod_bits,
                        left_neg_one,
                        right_neg_one,
                    ) {
                        continue;
                    }
                    SsaPointwise::fermat_basecase_mul_assign_left(
                        left,
                        right,
                        mod_bits,
                        *plan,
                        product_scratch,
                    );
                }
            }
        }
        PointwiseMulStrategy::Transform(plan) => {
            for index in 0..transform_len {
                // SAFETY: index is within bounds of transform_len complete coefficients.
                let left = unsafe { SsaTransform::coeff_mut(left_matrix, index, cl) };
                // SAFETY: same bounds proof as left, applied to the right matrix.
                let right = unsafe { SsaTransform::coeff_mut(right_matrix, index, cl) };

                // SAFETY: both coefficients and the complete nested arena are
                // disjoint, and the retained plan matches this modulus.
                unsafe {
                    let left_neg_one = SsaRing::normalize(left, mod_bits);
                    let right_neg_one = SsaRing::normalize(right, mod_bits);
                    if SsaPointwise::write_pointwise_special_product(
                        left,
                        right,
                        mod_bits,
                        left_neg_one,
                        right_neg_one,
                    ) {
                        continue;
                    }
                    plan.mul_assign_left(left, right, product_scratch);
                }
            }
        }
    }
}
