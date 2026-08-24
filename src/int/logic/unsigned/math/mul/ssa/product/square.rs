//! Pointwise transform squaring and coefficient traversal.

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    FftPlan, Limb, Multiplication, Residue, SSA_BASE_MODULUS_BITS, SquarePlan, SsaPlan,
    SsaPointwise, SsaRing, SsaTransform, TierCeiling,
};

impl SsaPointwise {
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
        // SAFETY: the caller guarantees dst spans cl limbs.
        unsafe { dst.get_unchecked_mut(..cl) }.fill(0);
        if class == Residue::NegOne {
            // SAFETY: cl is non-zero, so index 0 is in bounds.
            unsafe {
                *dst.get_unchecked_mut(0) = 1;
            }
        }
        true
    }

    /// Handles the pointwise special representation for squaring in place.
    ///
    /// Canonical `-1` (represented as `2^n` with guard = 1) squares to 1.
    ///
    /// # Safety
    /// `val` is a canonical `SsaRing::coeff_limbs(mod_bits)` span.
    #[allow(
        clippy::inline_always,
        reason = "one guard check replaces the full-width square"
    )]
    #[inline(always)]
    pub unsafe fn write_pointwise_special_square(val: &mut [Limb], mod_bits: usize) -> bool {
        let ml = SsaRing::mod_limbs(mod_bits);
        // SAFETY: val covers at least cl = ml + 1 limbs.
        let guard = unsafe { *val.get_unchecked(ml) };
        debug_assert!(guard <= 1, "pointwise input was canonicalized");
        if guard != 0 {
            let cl = ml.wrapping_add(1);
            // SAFETY: val covers at least cl limbs.
            let dest = unsafe { val.get_unchecked_mut(..cl) };
            dest.fill(0);
            // SAFETY: cl >= 2 > 0.
            unsafe {
                *dest.get_unchecked_mut(0) = 1;
            }
            return true;
        }
        false
    }

    /// Squares every coefficient of a transformed matrix in place.
    ///
    /// # Safety
    /// `matrix` holds `transform_len` complete `SsaRing::coeff_limbs(mod_bits)`-limb
    /// coefficients and `sqr_scratch` is sized from the same plan.
    pub unsafe fn pointwise_square_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        mod_bits: usize,
        executor: &E,
        sqr_scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(mod_bits);
        let basecase_plan = (mod_bits <= SSA_BASE_MODULUS_BITS)
            .then(|| Multiplication::select_square_plan(ml, TierCeiling::Full));
        // The nested geometry is a pure function of the ring width, so one
        // construction here keeps the per-coefficient loop below free of
        // planner searches.
        let nested_plan = (mod_bits > SSA_BASE_MODULUS_BITS).then(|| FftPlan::new(mod_bits));

        let cl = SsaRing::coeff_limbs(mod_bits);
        let coefficient_work = SsaPlan::basecase_product_cost(cl);
        if SsaTransform::has_parallel_work(transform_len, coefficient_work, executor) {
            let needed_scratch_result = if mod_bits <= SSA_BASE_MODULUS_BITS {
                cl.checked_add(Self::fermat_basecase_scratch_len(mod_bits))
            } else {
                cl.checked_add(nested_plan.map_or(0, |plan| plan.transform_sqr_scratch()))
            };
            let Some(needed_scratch) = needed_scratch_result else {
                debug_assert!(
                    false,
                    "pointwise square scratch size overflowed during preparation"
                );
                return;
            };
            let workers =
                FftPlan::pointwise_parallelism_budget(transform_len, executor.parallelism().get());
            let leaf_len = transform_len.div_ceil(workers).max(1);
            let leaf_count = FftPlan::pointwise_leaf_count(transform_len, leaf_len);
            let Some(required_scratch) = needed_scratch.checked_mul(leaf_count) else {
                debug_assert!(
                    false,
                    "pointwise square leaf arena size overflowed during preparation"
                );
                return;
            };
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
                    basecase_plan,
                    nested_plan,
                    needed_scratch,
                    leaf_len,
                    sqr_scratch,
                    executor,
                );
            }
            return;
        }

        // SAFETY: matrix and sqr_scratch are valid for transform_len coefficients.
        unsafe {
            pointwise_square_sequential(
                matrix,
                transform_len,
                mod_bits,
                basecase_plan,
                nested_plan,
                sqr_scratch,
            );
        }
    }
}

/// Executes pointwise squaring in disjoint coefficient ranges.
///
/// # Safety
/// The matrix contains `transform_len` complete coefficients. Recursive
/// splits occur only on coefficient boundaries; every leaf owns an independent
/// caller-provided scratch range before calling the shared sequential kernel.
#[allow(
    clippy::too_many_arguments,
    reason = "The recursive worker carries one immutable plan and one executor alongside the disjoint matrix ranges"
)]
unsafe fn pointwise_square_parallel<E: ParallelExecutor>(
    matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<SquarePlan>,
    nested_plan: Option<FftPlan>,
    needed_scratch: usize,
    leaf_len: usize,
    scratch: &mut [Limb],
    executor: &E,
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    if transform_len <= leaf_len {
        debug_assert!(
            scratch.len() >= needed_scratch,
            "validated pointwise square leaf scratch must cover its workspace"
        );
        // SAFETY: the outer preparation proved one complete arena per leaf.
        let leaf_scratch = unsafe { scratch.get_unchecked_mut(..needed_scratch) };
        // SAFETY: this leaf is a complete coefficient-aligned matrix partition.
        unsafe {
            pointwise_square_sequential(
                matrix,
                transform_len,
                mod_bits,
                basecase_plan,
                nested_plan,
                leaf_scratch,
            );
        }
        return;
    }

    let left_count = transform_len.div_euclid(2);
    let left_limbs = left_count.wrapping_mul(cl);
    let (left_matrix, right_matrix) = matrix.split_at_mut(left_limbs);
    let right_count = transform_len.wrapping_sub(left_count);
    let left_leaves = FftPlan::pointwise_leaf_count(left_count, leaf_len);
    let left_scratch_len = needed_scratch.wrapping_mul(left_leaves);
    debug_assert!(
        left_scratch_len != usize::MAX,
        "validated pointwise square leaf arena must fit the caller scratch"
    );
    let (left_scratch, right_scratch) = scratch.split_at_mut(left_scratch_len);
    let ((), ()) = executor.join(
        || {
            // SAFETY: the first matrix range is disjoint and complete.
            unsafe {
                pointwise_square_parallel(
                    left_matrix,
                    left_count,
                    mod_bits,
                    basecase_plan,
                    nested_plan,
                    needed_scratch,
                    leaf_len,
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
                    basecase_plan,
                    nested_plan,
                    needed_scratch,
                    leaf_len,
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
/// coefficients and `sqr_scratch` has at least `cl` limbs plus recursive scratch.
unsafe fn pointwise_square_sequential(
    matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<SquarePlan>,
    nested_plan: Option<FftPlan>,
    sqr_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    let (result, mul_scratch) = sqr_scratch.split_at_mut(cl);

    if let Some(plan) = basecase_plan {
        for index in 0..transform_len {
            // SAFETY: index is inside the complete coefficient matrix.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl) };
            // SAFETY: slot and scratch are complete disjoint coefficients.
            unsafe {
                SsaRing::normalize(slot, mod_bits);
                if SsaPointwise::write_pointwise_special_square(slot, mod_bits) {
                    continue;
                }
                SsaPointwise::fermat_basecase_sqr_assign(slot, mod_bits, plan, mul_scratch);
            }
        }
    } else {
        let Some(plan) = nested_plan else {
            debug_assert!(
                false,
                "the nested square loop must preselect its transform plan"
            );
            return;
        };
        for index in 0..transform_len {
            // SAFETY: index is inside the complete coefficient matrix.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, cl) };
            // SAFETY: slot and result are disjoint complete coefficients, and
            // the nested plan matches this modulus.
            unsafe {
                SsaRing::normalize(slot, mod_bits);
                if SsaPointwise::write_pointwise_special_square(slot, mod_bits) {
                    continue;
                }
                fermat_sqr_into(result, slot, mod_bits, &plan, mul_scratch);
                slot.copy_from_slice(result);
            }
        }
    }
}

/// Squares one Fermat residue with a preselected nested transform.
///
/// Recursion goes to the squaring driver, not the product one. An inner ring
/// above the basecase width has its own transform, and handing it to
/// `fft_mul_mod_slices_with_executor` with two aliased operands ran two forward
/// transforms over identical data — losing the square's discount at exactly
/// the widths where the pointwise stage nests, which since the inner-ring
/// rounding fix is every RAM-resident size.
///
/// # Safety
/// `dst` and `value` are disjoint complete coefficients, `plan` was built for
/// this exact `mod_bits`, and `mul_scratch` is sized for the nested transform.
unsafe fn fermat_sqr_into(
    dst: &mut [Limb],
    value: &[Limb],
    mod_bits: usize,
    plan: &FftPlan,
    mul_scratch: &mut [Limb],
) {
    // SAFETY: the caller provides disjoint complete coefficients, a plan for
    // this modulus, and the corresponding transform scratch.
    unsafe {
        SsaTransform::fft_sqr_mod_slices_with_executor(
            dst,
            value,
            mod_bits,
            false,
            Some(plan),
            &SequentialExecutor,
            mul_scratch,
        );
    }
}
