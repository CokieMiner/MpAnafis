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
    /// - `dst` has at least `SsaRing::coeff_limbs(modulus_bits)` limbs.
    /// - Both operands have the same layout: either each has that complete guarded width,
    ///   or each has exactly `SsaRing::mod_limbs(modulus_bits)` limbs with an implicit zero
    ///   guard and nonzero exact widths supplied through `significant_bits`.
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
            debug_assert_eq!(left.len(), ml, "short Fermat operand has exact data width");
            debug_assert_eq!(right.len(), ml, "short Fermat operand has exact data width");
            debug_assert!(
                significant_bits
                    .is_some_and(|(left_bits, right_bits)| { left_bits != 0 && right_bits != 0 }),
                "implicit-guard operands must carry nonzero exact widths"
            );
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            debug_assert!(
                guarded_operands,
                "implicit guards are only accepted by the transform path"
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

        let half_transform_bits = plan.chunk_bits.wrapping_mul(plan.transform_len >> 1);
        let (left_upper_half_zero, right_upper_half_zero) =
            significant_bits.map_or((false, false), |(left_bits, right_bits)| {
                (
                    left_bits <= half_transform_bits,
                    right_bits <= half_transform_bits,
                )
            });

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
        // child ranges.
        let (left_matrix, after_left) = scratch.split_at_mut(plan.mat_limbs);
        let (right_matrix, after_right) = after_left.split_at_mut(plan.mat_limbs);
        let (left_twiddle, after_left_twiddle) = after_right.split_at_mut(twiddle_len);
        let (right_twiddle, after_right_twiddle) = after_left_twiddle.split_at_mut(twiddle_len);
        let (product_scratch, recon_scratch) =
            after_right_twiddle.split_at_mut(plan.pointwise_scratch_for_parallelism(slots));
        left_twiddle.fill(0);
        right_twiddle.fill(0);

        // A whole-bit twist and stage 1 DIF butterfly can be applied while splitting,
        // so each matrix is written once directly with stage 1 completed.
        let fused_left_stage1 = if left_upper_half_zero {
            // SAFETY: left_matrix is partitioned with plan.mat_limbs and left_twiddle
            // has the plan's executor-specific twiddle arena.
            unsafe {
                SsaCoefficients::split_twisted_and_stage1_dif(
                    left,
                    left_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    plan.omega_shift,
                    left_twiddle,
                )
            }
        } else {
            false
        };
        if !fused_left_stage1 {
            // SAFETY: left_matrix has plan.mat_limbs and left_twiddle is disjoint.
            let fused_twist = unsafe {
                SsaCoefficients::split_twisted(
                    left,
                    left_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    left_twiddle,
                )
            };
            if !fused_twist {
                SsaCoefficients::split(
                    left,
                    left_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                );
                // SAFETY: left_matrix has plan.mat_limbs complete coefficients and
                // left_twiddle is disjoint.
                unsafe {
                    apply_forward_twist(left_matrix, None, &plan, left_twiddle);
                }
            }
        }

        let fused_right_stage1 = if right_upper_half_zero {
            // SAFETY: right_matrix is partitioned with plan.mat_limbs and right_twiddle
            // has the plan's executor-specific twiddle arena.
            unsafe {
                SsaCoefficients::split_twisted_and_stage1_dif(
                    right,
                    right_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    plan.omega_shift,
                    right_twiddle,
                )
            }
        } else {
            false
        };
        if !fused_right_stage1 {
            // SAFETY: right_matrix has plan.mat_limbs and right_twiddle is disjoint.
            let fused_twist = unsafe {
                SsaCoefficients::split_twisted(
                    right,
                    right_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    right_twiddle,
                )
            };
            if !fused_twist {
                SsaCoefficients::split(
                    right,
                    right_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                );
                // SAFETY: right_matrix has plan.mat_limbs complete coefficients and
                // right_twiddle is disjoint.
                unsafe {
                    apply_forward_twist(right_matrix, None, &plan, right_twiddle);
                }
            }
        }

        let ((), ()) = executor.join(
            || {
                // SAFETY: the left matrix and the first twiddle span are disjoint,
                // complete plan-sized partitions owned by this closure.
                unsafe {
                    run_forward_fft(
                        left_matrix,
                        left_twiddle,
                        &plan,
                        fused_left_stage1,
                        left_upper_half_zero,
                        executor,
                    );
                }
            },
            || {
                // SAFETY: the right matrix and its private twiddle span are disjoint,
                // complete plan-sized partitions owned by this closure.
                unsafe {
                    run_forward_fft(
                        right_matrix,
                        right_twiddle,
                        &plan,
                        fused_right_stage1,
                        right_upper_half_zero,
                        executor,
                    );
                }
            },
        );

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

        // SAFETY: matrices correctly sized, and the left twiddle arena is
        // disjoint and sized for the transform recursion.
        unsafe {
            Self::fft_in_place_with_executor(
                left_matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                false,
                executor,
                left_twiddle,
            );
        }

        // The inverse twiddle rides along with the reconstruction sweep, so the
        // coefficient matrix never makes a separate correction pass through RAM.
        SsaCoefficients::reconstruct(
            left_matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), left_twiddle)),
        );

        // SAFETY: dst has cl limbs, modulus_bits matches.
        unsafe {
            SsaRing::normalize(dst, modulus_bits);
        }
    }

    /// Core recursive FFT squaring using the supplied synchronous executor.
    ///
    /// The square pointwise pass remains sequential because its recursive product
    /// scratch is one shared buffer. The forward and inverse transform sweeps use
    /// the supplied executor and their own disjoint twiddle arena.
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
        let (matrix, after_matrix) = scratch.split_at_mut(plan.mat_limbs);
        let (twiddle_scratch, after_twiddle) = after_matrix.split_at_mut(twiddle_len);
        let (sqr_scratch, recon_scratch) =
            after_twiddle.split_at_mut(plan.square_scratch_for_slots(slots));
        twiddle_scratch.fill(0);

        let half_transform_bits = plan.chunk_bits.wrapping_mul(plan.transform_len >> 1);
        let a_upper_half_zero = a.len().wrapping_mul(LIMB_BITS) <= half_transform_bits;

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
            // SAFETY: matrix and twiddle scratch are complete, disjoint plan-sized spans.
            let fused_twist = unsafe {
                SsaCoefficients::split_twisted(
                    a,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.twist_step_half,
                    twiddle_scratch,
                )
            };
            if !fused_twist {
                SsaCoefficients::split(
                    a,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                );
                // SAFETY: matrix and twiddle scratch are complete and disjoint.
                unsafe {
                    apply_forward_twist(matrix, None, &plan, twiddle_scratch);
                }
            }
        }

        // SAFETY: all matrix and scratch spans were partitioned from the plan.
        unsafe {
            run_forward_fft(
                matrix,
                twiddle_scratch,
                &plan,
                fused_stage1,
                a_upper_half_zero,
                executor,
            );
            SsaPointwise::pointwise_square_with_executor(
                matrix,
                plan.transform_len,
                plan.inner_bits,
                executor,
                sqr_scratch,
            );
            Self::fft_in_place_with_executor(
                matrix,
                plan.transform_len,
                plan.omega_shift,
                plan.inner_bits,
                true,
                false,
                executor,
                twiddle_scratch,
            );
        }

        // The inverse twiddle rides along with the reconstruction sweep, so the
        // coefficient matrix never makes a separate correction pass through RAM.
        SsaCoefficients::reconstruct(
            matrix,
            plan.transform_len,
            plan.chunk_bits,
            plan.inner_bits,
            modulus_bits,
            dst,
            recon_scratch,
            Some((plan.inverse_twist(), twiddle_scratch)),
        );
        // SAFETY: dst has cl limbs and modulus_bits matches.
        unsafe {
            SsaRing::normalize(dst, modulus_bits);
        }
    }
}

/// Applies the forward pre-twist to one or more coefficient matrices.
/// Accumulating the half-bit exponent inside one ring period prevents wrapping.
///
/// # Safety
/// Every matrix is complete; `twiddle` is a disjoint `inner_cl`-limb buffer.
unsafe fn apply_forward_twist(
    matrix: &mut [Limb],
    second_matrix: Option<&mut [Limb]>,
    plan: &FftPlan,
    twiddle: &mut [Limb],
) {
    let period = plan.inner_bits.wrapping_mul(4);
    let mut shift = 0_usize;
    if let Some(second) = second_matrix {
        for index in 1_usize..plan.transform_len {
            shift = shift.wrapping_add(plan.twist_step_half);
            if shift >= period {
                shift = shift.wrapping_sub(period);
            }
            // SAFETY: index < transform_len and matrix holds complete coefficients.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, plan.inner_cl) };
            // SAFETY: second has the same complete, disjoint layout.
            let second_slot = unsafe { SsaTransform::coeff_mut(second, index, plan.inner_cl) };
            // SAFETY: slots are canonical and disjoint from twiddle.
            unsafe {
                SsaRing::shift_half(slot, shift, plan.inner_bits, twiddle);
                SsaRing::shift_half(second_slot, shift, plan.inner_bits, twiddle);
            }
        }
    } else {
        for index in 1_usize..plan.transform_len {
            shift = shift.wrapping_add(plan.twist_step_half);
            if shift >= period {
                shift = shift.wrapping_sub(period);
            }
            // SAFETY: index < transform_len and matrix holds complete coefficients.
            let slot = unsafe { SsaTransform::coeff_mut(matrix, index, plan.inner_cl) };
            // SAFETY: slot is canonical and disjoint from twiddle after splitting.
            unsafe {
                SsaRing::shift_half(slot, shift, plan.inner_bits, twiddle);
            }
        }
    }
}

// ── Private helper functions ─────────────────────────────────────────────────

/// Runs one complete forward FFT after operand staging.
///
/// # Safety
/// `matrix` is a complete transform matrix and `twiddle` is a disjoint scratch
/// span sized for `plan`; the fused-stage flag and upper-half flag describe the
/// exact staging operation that produced `matrix`.
unsafe fn run_forward_fft<E: ParallelExecutor>(
    matrix: &mut [Limb],
    twiddle: &mut [Limb],
    plan: &FftPlan,
    fused_stage1: bool,
    upper_half_zero: bool,
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
                upper_half_zero,
                executor,
                twiddle,
            );
        }
    }
}
