//! End-to-end multiplication and squaring orchestration for SSA.

use crate::parallel::ParallelExecutor;

use super::{
    FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SsaCoefficients, SsaPointwise, SsaRing,
};

/// Namespace for SSA transform operations and layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaTransform;

/// The complete multiply and square transforms, and the operand measurement
/// their ring sizing starts from.
///
/// Other transform files contribute their operations to this namespace.
impl SsaTransform {
    /// Core recursive FFT multiplication in `Z/(2^modulus_bits + 1)`.
    ///
    /// A `forced_plan` overrides the planner's geometry. It is passed as a built
    /// plan rather than as a bare exponent so that the only way to reach this
    /// function with a forced geometry is to have already constructed a valid one
    /// for `modulus_bits` — an exponent that does not divide this ring width simply
    /// cannot be expressed here.
    ///
    /// The executor only forks disjoint matrix and twiddle-scratch borrows, so
    /// callers can select sequential, Rayon, or an application-owned policy.
    ///
    /// # Safety
    /// - `dst` either has at least `SsaRing::coeff_limbs(modulus_bits)` limbs, or
    ///   has at most `SsaRing::mod_limbs(modulus_bits)` limbs and the sum of the
    ///   supplied significant widths fits its exact, guard-free output span.
    /// - Both operands have the same layout: either each has that complete guarded width,
    ///   or each has at most `SsaRing::mod_limbs(modulus_bits)` limbs with implicit zero
    ///   high limbs and guard, plus nonzero exact widths supplied through `significant_bits`.
    /// - Values in `significant_bits`, when present, are upper bounds on the
    ///   represented operands' exact significant widths.
    /// - `forced_plan`, when present, was built for this exact `modulus_bits`, and
    ///   `scratch` is sized from it.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "FFT orchestration linear pass"
    )]
    pub unsafe fn fft_mul_mod_slices_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        left: &[Limb],
        right: &[Limb],
        modulus_bits: usize,
        significant_bits: Option<(usize, usize)>,
        force_transform: bool,
        forced_plan: Option<&FftPlan>,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let output_has_guard = dst.len() > ml;
        debug_assert!(
            output_has_guard
                || significant_bits.is_some_and(|(left_bits, right_bits)| {
                    let product_bits = left_bits.saturating_add(right_bits);
                    product_bits <= modulus_bits
                        && product_bits <= dst.len().saturating_mul(LIMB_BITS)
                }),
            "a guard-free Fermat output must hold the proven exact product"
        );
        let left_guarded = left.len() > ml;
        let right_guarded = right.len() > ml;
        debug_assert_eq!(
            left_guarded, right_guarded,
            "SSA operands must both include the guard limb or both omit it"
        );
        let guarded_operands = left_guarded;
        if guarded_operands {
            // SAFETY: both operands include the complete guard limb in this branch.
            if unsafe {
                SsaPointwise::write_special_residue_product(dst, left, right, modulus_bits)
            } {
                return;
            }
        } else {
            debug_assert!(left.len() <= ml, "short Fermat operand fits the data width");
            debug_assert!(
                right.len() <= ml,
                "short Fermat operand fits the data width"
            );
            debug_assert!(
                significant_bits.is_some_and(|(left_bits, right_bits)| {
                    left_bits != 0
                        && right_bits != 0
                        && left_bits <= left.len().saturating_mul(LIMB_BITS)
                        && right_bits <= right.len().saturating_mul(LIMB_BITS)
                }),
                "implicit-zero operands must carry valid nonzero exact widths"
            );
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            debug_assert!(
                guarded_operands && output_has_guard,
                "implicit guards and exact outputs are only accepted by the transform path"
            );
            let bc_len = SsaPointwise::fermat_basecase_scratch_len(modulus_bits);
            let (bc, _) = scratch.split_at_mut(bc_len);
            // SAFETY: all three coefficients have cl limbs, not -1 or zero.
            unsafe {
                SsaPointwise::fermat_basecase_mul_into(dst, left, right, modulus_bits, bc);
            }
            return;
        }

        // ── Recursive FFT case ───────────────────────────────────────────────
        let plan = forced_plan
            .copied()
            .unwrap_or_else(|| FftPlan::new(modulus_bits));

        let (active_left_chunks, active_right_chunks) = significant_bits.map_or(
            (plan.transform_len, plan.transform_len),
            |(left_bits, right_bits)| {
                (
                    left_bits.div_ceil(plan.chunk_bits).min(plan.transform_len),
                    right_bits.div_ceil(plan.chunk_bits).min(plan.transform_len),
                )
            },
        );

        let half_len = plan.transform_len >> 1;
        let left_upper_half_zero = active_left_chunks <= half_len;
        let right_upper_half_zero = active_right_chunks <= half_len;

        // This point is past the basecase early return, so the transform layout is
        // the one that must fit regardless of how narrow the ring is. Every caller
        // sizes its buffer from the same plan, so this never re-allocates; the
        // assertion catches a mis-sized caller in debug and test builds rather than
        // hiding an allocation in the hot path.
        let slots = plan.parallel_slots(executor.parallelism().get());
        let Some(twiddle_len) = plan.inner_cl.checked_mul(slots) else {
            debug_assert!(false, "validated twiddle arena size overflowed");
            return;
        };
        debug_assert!(
            scratch.len() >= plan.transform_mul_scratch_for_slots(slots),
            "SSA transform scratch is undersized: mod {}, inner {}, len {}, have {}, need {}, slots {}",
            plan.modulus_bits,
            plan.inner_bits,
            plan.transform_len,
            scratch.len(),
            plan.transform_mul_scratch_for_slots(slots),
            slots
        );

        // Partition caller scratch: [left] [right] [left twiddles] [right twiddles]
        // [pointwise leaf arenas] [reconstruction]. Each twiddle and pointwise
        // arena is divided by its recursion into disjoint coefficient-aligned
        // child ranges. Every arena is pure staging — each consumer writes its
        // span before reading it — so none needs a zeroing pass.
        let (left_matrix, after_left) = scratch.split_at_mut(plan.mat_limbs);
        let (right_matrix, after_right) = after_left.split_at_mut(plan.mat_limbs);
        let (left_twiddle, after_left_twiddle) = after_right.split_at_mut(twiddle_len);
        let (right_twiddle, after_right_twiddle) = after_left_twiddle.split_at_mut(twiddle_len);
        let (product_scratch, recon_scratch) =
            after_right_twiddle.split_at_mut(plan.pointwise_scratch_for_parallelism(slots));

        // Stage each operand and run its forward FFT as one unit, so a
        // parallel executor forks the split sweeps together with the
        // transforms instead of paying both serially before the join. Each
        // branch owns a disjoint matrix and twiddle arena, and the pre-twist
        // (whole- or half-bit step) still fuses with stage 1 DIF whenever the
        // active chunks fit the matrix's lower half.
        if executor.parallelism().get() > 1 {
            let ((), ()) = executor.join(
                || {
                    // SAFETY: the left matrix and first twiddle span are disjoint,
                    // complete plan-sized partitions owned by this closure.
                    unsafe {
                        stage_and_run_forward_fft(
                            left,
                            left_matrix,
                            left_twiddle,
                            &plan,
                            left_upper_half_zero,
                            active_left_chunks,
                            executor,
                        );
                    }
                },
                || {
                    // SAFETY: the right matrix and private twiddle span are disjoint,
                    // complete plan-sized partitions owned by this closure.
                    unsafe {
                        stage_and_run_forward_fft(
                            right,
                            right_matrix,
                            right_twiddle,
                            &plan,
                            right_upper_half_zero,
                            active_right_chunks,
                            executor,
                        );
                    }
                },
            );
        } else {
            // SAFETY: each matrix and its private twiddle arena is complete;
            // sequential execution permits reusing the caller thread directly.
            unsafe {
                stage_and_run_forward_fft(
                    left,
                    left_matrix,
                    left_twiddle,
                    &plan,
                    left_upper_half_zero,
                    active_left_chunks,
                    executor,
                );
                stage_and_run_forward_fft(
                    right,
                    right_matrix,
                    right_twiddle,
                    &plan,
                    right_upper_half_zero,
                    active_right_chunks,
                    executor,
                );
            }
        }

        // ── Pointwise products ───────────────────────────────────────────────
        // SAFETY: the matrices are perfectly sized for the complete transform.
        unsafe {
            SsaPointwise::pointwise_multiply_with_executor(
                left_matrix,
                right_matrix,
                plan.transform_len,
                plan.inner_bits,
                executor,
                product_scratch,
            );
        }

        let active_out_chunks =
            significant_bits.map_or(plan.transform_len, |(left_bits, right_bits)| {
                left_bits
                    .saturating_add(right_bits)
                    .div_ceil(plan.chunk_bits)
                    .min(plan.transform_len)
            });

        // SAFETY: matrices correctly sized, and the left twiddle arena is
        // disjoint and sized for the transform recursion.
        unsafe {
            Self::fft_in_place_with_executor(
                left_matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                active_out_chunks,
                executor,
                left_twiddle,
            );
        }

        // The inverse twiddle runs as its own sweep before accumulation — the
        // same number of coefficient touches, but each coefficient's shift is
        // independent, so a parallel executor can fork the sweep.
        SsaCoefficients::reconstruct(
            left_matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), left_twiddle)),
            executor,
        );

        if output_has_guard {
            // SAFETY: output_has_guard proves dst contains the data width and
            // guard limb required by this ring.
            unsafe {
                SsaRing::normalize(dst, modulus_bits);
            }
        }
    }

    /// Core recursive FFT squaring using the supplied synchronous executor.
    ///
    /// The pointwise squares fork over disjoint coefficient ranges, each with
    /// its own arena; nested coefficient squares run sequentially under a
    /// child executor so they cannot oversubscribe the outer one. The forward
    /// and inverse transform sweeps use the supplied executor and their own
    /// disjoint twiddle arena.
    ///
    /// # Safety
    /// `dst` and `a` have complete coefficient widths for `modulus_bits`, and
    /// `scratch` is sized from the selected plan and executor parallelism.
    /// When supplied, `forced_plan` was built for this exact modulus width and
    /// its executor-sized scratch geometry.
    #[allow(
        clippy::too_many_lines,
        reason = "FFT squaring orchestration linear pass"
    )]
    pub unsafe fn fft_sqr_mod_slices_with_executor<E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        modulus_bits: usize,
        force_transform: bool,
        forced_plan: Option<&FftPlan>,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let cl = SsaRing::coeff_limbs(modulus_bits);
        let guarded = a.len() > ml;
        if guarded {
            // SAFETY: a has at least cl = ml + 1 limbs.
            let class = unsafe { SsaRing::classify_residue(a, ml) };
            // SAFETY: dst has cl limbs and cl = ml + 1 is non-zero.
            if unsafe { SsaPointwise::write_special_residue_square(dst, cl, class) } {
                return;
            }
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            // SAFETY: coefficient is canonical, not -1 or zero, buffers correctly sized.
            unsafe {
                SsaPointwise::fermat_basecase_sqr_into(dst, a, modulus_bits, scratch);
            }
            return;
        }

        let plan = forced_plan
            .copied()
            .unwrap_or_else(|| FftPlan::new(modulus_bits));
        let slots = plan.parallel_slots(executor.parallelism().get());
        let Some(twiddle_len) = plan.inner_cl.checked_mul(slots) else {
            debug_assert!(false, "validated square twiddle arena size overflowed");
            return;
        };
        debug_assert!(
            scratch.len() >= plan.transform_sqr_scratch_for_slots(slots),
            "SSA square scratch is undersized for the executor's twiddle slots"
        );

        // Partition caller scratch: [matrix] [twiddles] [square] [reconstruction].
        // The twiddle arena is pure staging — every consumer writes its span
        // before reading it — so it needs no zeroing pass.
        let (matrix, after_matrix) = scratch.split_at_mut(plan.mat_limbs);
        let (twiddle_scratch, after_twiddle) = after_matrix.split_at_mut(twiddle_len);
        let (sqr_scratch, recon_scratch) =
            after_twiddle.split_at_mut(plan.pointwise_scratch_for_parallelism(slots));

        // The guard limb carries no chunk data, so the active chunk count comes
        // from the data width alone.
        let active_chunks = ml
            .wrapping_mul(LIMB_BITS)
            .div_ceil(plan.chunk_bits)
            .min(plan.transform_len);
        let a_upper_half_zero = active_chunks <= (plan.transform_len >> 1);

        // A whole-bit twist and stage 1 DIF butterfly can be applied while splitting,
        // so the matrix is written once directly with stage 1 completed.
        let fused_stage1 = if a_upper_half_zero {
            // SAFETY: matrix is partitioned with plan.mat_limbs and twiddle_scratch
            // has the plan's executor-specific twiddle arena.
            unsafe {
                SsaCoefficients::split_twisted_and_stage1_dif(
                    a,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    plan.omega_shift,
                    twiddle_scratch,
                )
            }
        } else {
            false
        };
        if !fused_stage1 {
            // The fused split handles every half-bit step, including the odd
            // steps that carry a sqrt(2) factor.
            // SAFETY: matrix has plan.mat_limbs and twiddle_scratch is a
            // disjoint two-coefficient arena.
            unsafe {
                SsaCoefficients::split_twisted(
                    a,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    twiddle_scratch,
                );
            }
        }

        // SAFETY: all matrix and scratch spans were partitioned from the plan.
        unsafe {
            run_forward_fft(
                matrix,
                twiddle_scratch,
                &plan,
                fused_stage1,
                active_chunks,
                executor,
            );
            SsaPointwise::pointwise_square_with_executor(
                matrix,
                plan.transform_len,
                plan.inner_bits,
                executor,
                sqr_scratch,
            );
            let active_out_chunks = (ml.wrapping_mul(LIMB_BITS).saturating_mul(2))
                .div_ceil(plan.chunk_bits)
                .min(plan.transform_len);
            Self::fft_in_place_with_executor(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                active_out_chunks,
                executor,
                twiddle_scratch,
            );
        }

        // The inverse twiddle runs as its own sweep before accumulation — the
        // same number of coefficient touches, but each coefficient's shift is
        // independent, so a parallel executor can fork the sweep.
        SsaCoefficients::reconstruct(
            matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), twiddle_scratch)),
            executor,
        );
        // SAFETY: dst has cl limbs and modulus_bits matches.
        unsafe {
            SsaRing::normalize(dst, modulus_bits);
        }
    }
}

// ── Private helper functions ─────────────────────────────────────────────────

/// Runs one complete forward FFT after operand staging.
///
/// # Safety
/// `matrix` is a complete transform matrix and `twiddle` is a disjoint scratch
/// span sized for `plan`; the fused-stage flag and active chunks describe the
/// exact staging operation that produced `matrix`.
unsafe fn run_forward_fft<E: ParallelExecutor>(
    matrix: &mut [Limb],
    twiddle: &mut [Limb],
    plan: &FftPlan,
    fused_stage1: bool,
    active_chunks: usize,
    executor: &E,
) {
    if fused_stage1 {
        // SAFETY: the caller proves the complete matrix and disjoint twiddle span.
        unsafe {
            SsaTransform::fft_in_place_from_stage2_with_executor(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                active_chunks,
                executor,
                twiddle,
            );
        }
    } else {
        // SAFETY: the caller proves the complete matrix and disjoint twiddle span.
        unsafe {
            SsaTransform::fft_in_place_with_executor(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                false,
                active_chunks,
                executor,
                twiddle,
            );
        }
    }
}

/// Stages one operand into its coefficient matrix, then runs its forward FFT.
///
/// Fuses the pre-twist with the first DIF butterfly stage whenever the
/// operand's active chunks fit the matrix's lower half; otherwise splits with
/// the twist alone and lets the transform run from stage 1.
///
/// # Safety
/// `matrix` and `twiddle` are disjoint complete plan-sized partitions, and
/// `src` is the operand the plan's geometry was selected for.
unsafe fn stage_and_run_forward_fft<E: ParallelExecutor>(
    src: &[Limb],
    matrix: &mut [Limb],
    twiddle: &mut [Limb],
    plan: &FftPlan,
    upper_half_zero: bool,
    active_chunks: usize,
    executor: &E,
) {
    let fused_stage1 = if upper_half_zero {
        // SAFETY: matrix is partitioned with plan.mat_limbs and twiddle has
        // the plan's executor-specific twiddle arena.
        unsafe {
            SsaCoefficients::split_twisted_and_stage1_dif(
                src,
                matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.twist_step_half,
                plan.omega_shift,
                twiddle,
            )
        }
    } else {
        false
    };
    if !fused_stage1 {
        // The fused split handles every half-bit step, including the odd
        // steps that carry a sqrt(2) factor.
        // SAFETY: matrix has plan.mat_limbs and twiddle is a disjoint
        // two-coefficient arena.
        unsafe {
            SsaCoefficients::split_twisted(
                src,
                matrix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.twist_step_half,
                twiddle,
            );
        }
    }
    // SAFETY: the matrix and its disjoint twiddle span are complete for the
    // plan, and the staging flag matches the split just performed.
    unsafe {
        run_forward_fft(matrix, twiddle, plan, fused_stage1, active_chunks, executor);
    }
}
