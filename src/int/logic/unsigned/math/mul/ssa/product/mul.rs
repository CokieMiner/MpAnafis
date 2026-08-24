//! Pointwise transform multiplication and coefficient traversal.

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    FftPlan, Limb, MulPlan, Multiplication, NegacyclicPlan, Residue, SSA_BASE_MODULUS_BITS,
    SsaPlan, SsaPointwise, SsaRing, SsaTransform, TierCeiling,
};

impl SsaPointwise {
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
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        mod_bits: usize,
    ) -> bool {
        let ml = SsaRing::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);
        // SAFETY: ml < cl and the caller guarantees both spans have cl limbs.
        let left_class = unsafe { SsaRing::classify_residue(left, ml) };
        // SAFETY: same bounds proof as left.
        let right_class = unsafe { SsaRing::classify_residue(right, ml) };

        if left_class == Residue::Zero || right_class == Residue::Zero {
            // SAFETY: the caller guarantees dst spans cl limbs. A slice fill
            // lowers to one memset; the iterator form did not.
            unsafe { dst.get_unchecked_mut(..cl) }.fill(0);
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
        destination.copy_from_slice(source);
        // SAFETY: dst has cl limbs and mod_bits matches.
        unsafe {
            SsaRing::negate(dst, mod_bits);
        }
        true
    }

    /// Multiplies every coefficient pair using the supplied synchronous executor.
    ///
    /// Both sequential and parallel paths use the caller-provided scratch.
    /// Parallel leaves receive disjoint coefficient-aligned regions; nested
    /// coefficient products use a sequential child executor to avoid recursive
    /// oversubscription.
    ///
    /// # Safety
    /// The caller must satisfy the validated matrix and scratch preconditions
    /// described above.
    pub unsafe fn pointwise_multiply_with_executor<E: ParallelExecutor>(
        left_matrix: &mut [Limb],
        right_matrix: &mut [Limb],
        transform_len: usize,
        mod_bits: usize,
        executor: &E,
        product_scratch: &mut [Limb],
    ) {
        let basecase_plan = (mod_bits <= SSA_BASE_MODULUS_BITS).then(|| {
            let ml = SsaRing::mod_limbs(mod_bits);
            Multiplication::select_plan(ml, ml, TierCeiling::Full)
        });
        let negacyclic_plan = (mod_bits <= SSA_BASE_MODULUS_BITS)
            .then(|| NegacyclicPlan::new(SsaRing::mod_limbs(mod_bits)))
            .flatten();
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
                cl.checked_add(nested_plan.map_or(0, |plan| plan.transform_mul_scratch()))
            };
            let Some(needed_scratch) = needed_scratch_result else {
                debug_assert!(
                    false,
                    "pointwise scratch size overflowed during preparation"
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
                    "pointwise leaf arena size overflowed during preparation"
                );
                return;
            };
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
                    basecase_plan,
                    negacyclic_plan,
                    nested_plan,
                    needed_scratch,
                    leaf_len,
                    product_scratch,
                    executor,
                );
            }
            return;
        }

        // SAFETY: left_matrix and right_matrix are disjoint slices.
        unsafe {
            pointwise_multiply_sequential(
                left_matrix,
                right_matrix,
                transform_len,
                mod_bits,
                basecase_plan,
                negacyclic_plan,
                nested_plan,
                product_scratch,
            );
        }
    }
}

/// Handles the sole pointwise special representation that cannot flow through
/// an ordinary fixed-width multiplication.
///
/// Canonical zero has a zero guard and needs no branch: every multiplication
/// tier naturally writes a zero product. Canonical `-1` is the unique value
/// with guard one, so it is identified without scanning the data limbs.
///
/// # Safety
/// `left` and `right` are disjoint, canonical `SsaRing::coeff_limbs(mod_bits)` spans.
#[allow(
    clippy::inline_always,
    reason = "one guard check per coefficient replaces two full-width classification scans"
)]
#[inline(always)]
unsafe fn write_pointwise_special_product(
    left: &mut [Limb],
    right: &[Limb],
    mod_bits: usize,
) -> bool {
    let ml = SsaRing::mod_limbs(mod_bits);
    // SAFETY: both complete coefficients include the guard at ml.
    let left_guard = unsafe { *left.get_unchecked(ml) };
    // SAFETY: same coefficient-width proof as left.
    let right_guard = unsafe { *right.get_unchecked(ml) };
    debug_assert!(
        left_guard <= 1 && right_guard <= 1,
        "pointwise inputs were canonicalized"
    );

    if left_guard != 0 {
        let cl = ml.wrapping_add(1);
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
    if right_guard != 0 {
        // SAFETY: left is canonical, so in-place negation is valid.
        unsafe {
            SsaRing::negate(left, mod_bits);
        }
        return true;
    }
    false
}

/// Multiplies two Fermat-ring elements with a preselected nested transform.
///
/// The ring is wider than `SSA_BASE_MODULUS_BITS`, so the product recurses into
/// the FFT. `plan` is the geometry selected once for every coefficient, keeping
/// planner searches out of the pointwise loop.
///
/// # Safety
/// - `dst`, `left`, and `right` are mutually disjoint complete coefficients.
/// - `plan` was built for this exact `mod_bits`.
/// - `mul_scratch` has at least the nested transform's scratch width.
#[allow(
    clippy::inline_always,
    reason = "benchmarking shows the fixed pointwise plan must propagate through the coefficient loop"
)]
#[inline(always)]
unsafe fn fermat_mul_into(
    dst: &mut [Limb],
    left: &[Limb],
    right: &[Limb],
    mod_bits: usize,
    plan: &FftPlan,
    mul_scratch: &mut [Limb],
) {
    // SAFETY: the caller provides three disjoint complete coefficients, a plan
    // for this modulus, and the corresponding transform scratch.
    unsafe {
        SsaTransform::fft_mul_mod_slices_with_executor(
            dst,
            left,
            right,
            mod_bits,
            None,
            false,
            Some(plan),
            &SequentialExecutor,
            mul_scratch,
        );
    }
}

/// Sequential loop for pointwise multiplication over a contiguous chunk of coefficients.
///
/// # Safety
/// - `left_matrix` and `right_matrix` each have `transform_len * SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `product_scratch` has at least `cl` limbs plus recursive scratch.
#[allow(
    clippy::too_many_arguments,
    reason = "the leaf worker carries the preselected plans alongside its matrix ranges"
)]
unsafe fn pointwise_multiply_sequential(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<MulPlan>,
    negacyclic_plan: Option<NegacyclicPlan>,
    nested_plan: Option<FftPlan>,
    product_scratch: &mut [Limb],
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    let (result, mul_scratch) = product_scratch.split_at_mut(cl);

    if let Some(plan) = negacyclic_plan {
        for i in 0..transform_len {
            let offset = i.wrapping_mul(cl);
            // SAFETY: offset + cl <= matrix length by construction; both matrices
            // are disjoint and each holds transform_len complete coefficients.
            let left = unsafe { left_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
            // SAFETY: same bounds proof as left, applied to the right matrix.
            let right = unsafe { right_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };

            // SAFETY: both spans are complete cl-limb coefficients.
            unsafe {
                SsaRing::normalize(left, mod_bits);
                SsaRing::normalize(right, mod_bits);
                if write_pointwise_special_product(left, right, mod_bits) {
                    continue;
                }
                plan.mul_assign_left(left, right, mul_scratch);
            }
        }
    } else if let Some(plan) = basecase_plan {
        for i in 0..transform_len {
            let offset = i.wrapping_mul(cl);
            // SAFETY: offset + cl <= matrix length by construction; both matrices
            // are disjoint and each holds transform_len complete coefficients.
            let left = unsafe { left_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
            // SAFETY: same bounds proof as left, applied to the right matrix.
            let right = unsafe { right_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };

            // SAFETY: both spans are complete cl-limb coefficients.
            unsafe {
                SsaRing::normalize(left, mod_bits);
                SsaRing::normalize(right, mod_bits);
                if write_pointwise_special_product(left, right, mod_bits) {
                    continue;
                }
                SsaPointwise::fermat_basecase_mul_assign_left(
                    left,
                    right,
                    mod_bits,
                    plan,
                    mul_scratch,
                );
            }
        }
    } else {
        let Some(plan) = nested_plan else {
            debug_assert!(
                false,
                "the nested pointwise loop must preselect its transform plan"
            );
            return;
        };
        for i in 0..transform_len {
            let offset = i.wrapping_mul(cl);
            // SAFETY: offset + cl <= matrix length by construction; both matrices
            // are disjoint and each holds transform_len complete coefficients.
            let left = unsafe { left_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
            // SAFETY: same bounds proof as left, applied to the right matrix.
            let right = unsafe { right_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };

            // SAFETY: the two operands and result scratch are disjoint complete
            // coefficients, and the nested plan matches this modulus.
            unsafe {
                SsaRing::normalize(left, mod_bits);
                SsaRing::normalize(right, mod_bits);
                if write_pointwise_special_product(left, right, mod_bits) {
                    continue;
                }
                fermat_mul_into(result, left, right, mod_bits, &plan, mul_scratch);
                left.copy_from_slice(result);
            }
        }
    }
}

/// Executes pointwise products in disjoint coefficient ranges.
///
/// # Safety
/// The two matrices contain `transform_len` complete coefficients. Recursive
/// splits occur only on coefficient boundaries; every leaf owns an independent
/// caller-provided scratch range before calling the shared sequential kernel.
#[allow(
    clippy::too_many_arguments,
    reason = "The recursive worker carries one immutable plan and one executor alongside the two disjoint matrix ranges"
)]
unsafe fn pointwise_multiply_parallel<E: ParallelExecutor>(
    left_matrix: &mut [Limb],
    right_matrix: &mut [Limb],
    transform_len: usize,
    mod_bits: usize,
    basecase_plan: Option<MulPlan>,
    negacyclic_plan: Option<NegacyclicPlan>,
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
                basecase_plan,
                negacyclic_plan,
                nested_plan,
                leaf_scratch,
            );
        }
        return;
    }

    let left_count = transform_len.div_euclid(2);
    // Matrix validity proves every coefficient boundary is representable.
    let left_limbs = left_count.wrapping_mul(cl);
    let (left_first, left_second) = left_matrix.split_at_mut(left_limbs);
    let (right_first, right_second) = right_matrix.split_at_mut(left_limbs);
    // `transform_len > leaf_len >= 1` proves the split has at least two
    // coefficients, so this subtraction cannot underflow.
    let right_count = transform_len.wrapping_sub(left_count);
    let left_leaves = FftPlan::pointwise_leaf_count(left_count, leaf_len);
    // The outer preparation checked the complete leaf arena before recursion.
    let left_scratch_len = needed_scratch.wrapping_mul(left_leaves);
    debug_assert!(
        left_scratch_len != usize::MAX,
        "validated pointwise leaf arena must fit the caller scratch"
    );
    let (left_scratch, right_scratch) = scratch.split_at_mut(left_scratch_len);
    let ((), ()) = executor.join(
        || {
            // SAFETY: the first matrix ranges are disjoint and complete.
            unsafe {
                pointwise_multiply_parallel(
                    left_first,
                    right_first,
                    left_count,
                    mod_bits,
                    basecase_plan,
                    negacyclic_plan,
                    nested_plan,
                    needed_scratch,
                    leaf_len,
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
                    basecase_plan,
                    negacyclic_plan,
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
