//! Complete and truncated multiplication-transform execution for SSA.

#![expect(
    unsafe_code,
    reason = "Operand-bound plans establish coefficient supports and reserve every matrix, twiddle, and reconstruction partition"
)]

use crate::parallel::ParallelExecutor;

use super::{
    DenseWorkspace, LIMB_BITS, Limb, LimbOutput, MulTransformInput, SSA_BASE_MODULUS_BITS,
    SsaCoefficients, SsaPointwise, SsaRing, TruncatedTransform,
};

/// Namespace for SSA transform operations and layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaTransform;

impl SsaTransform {
    /// Core recursive FFT multiplication in `Z/(2^modulus_bits + 1)`.
    ///
    /// A constructed `forced_plan` replaces the selected geometry for this ring.
    /// Each concurrent phase owns disjoint matrix and workspace partitions.
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
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "FFT orchestration linear pass"
    )]
    pub unsafe fn fft_mul_mod_slices_with_executor<'plan, E: ParallelExecutor>(
        dst: &mut [impl LimbOutput],
        left: &[Limb],
        right: &[Limb],
        modulus_bits: usize,
        significant_bits: Option<(usize, usize)>,
        force_transform: bool,
        forced_plan: impl Into<MulTransformInput<'plan>>,
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
            // SAFETY: all three coefficients have cl limbs, not -1 or zero.
            // The caller's arena already covers the selected basecase.
            unsafe {
                SsaPointwise::fermat_basecase_mul_into(dst, left, right, modulus_bits, scratch);
            }
            return;
        }

        // Unforced multiplication falls back to the product-optimal geometry.
        let plan = forced_plan.into().resolve(modulus_bits, false);
        let leaf_budget = plan.pointwise_leaf_count(executor.parallelism().get());

        let (active_left_chunks, active_right_chunks) = significant_bits.map_or(
            (plan.transform_len, plan.transform_len),
            |(left_bits, right_bits)| {
                (
                    left_bits
                        .div_ceil(plan.chunk_bits.get())
                        .min(plan.transform_len),
                    right_bits
                        .div_ceil(plan.chunk_bits.get())
                        .min(plan.transform_len),
                )
            },
        );

        let support =
            Self::product_support(active_left_chunks, active_right_chunks, plan.transform_len);
        if support < plan.transform_len {
            let transform = TruncatedTransform::new(&plan);
            let row_log = plan.transform_len.trailing_zeros() >> 1;
            // SAFETY: row_log=floor(log2(K)/2)<=log2(K).
            let column_log = unsafe { plan.transform_len.trailing_zeros().unchecked_sub(row_log) };
            let n1 = support >> column_log;
            let workers = executor.parallelism().get();
            if support > (plan.transform_len >> 1)
                && plan.transform_len >= 4
                && plan.transform_len > transform.max_resident
                && n1 >= workers
            {
                // SAFETY: K>=4, K/2<support<K, and significant widths prove a
                // zero product tail above support. The constructed geometry and
                // accepted arena supply complete matrices, row work, and
                // reconstruction storage; dst holds the guarded or exact product.
                unsafe {
                    Self::fused_matrix_truncated_mul(
                        left,
                        right,
                        dst,
                        active_left_chunks,
                        active_right_chunks,
                        support,
                        &plan,
                        &transform,
                        executor,
                        scratch,
                    );
                }
                return;
            }
            // SAFETY: significant widths bound the input polynomials; their
            // product has degree below support with no negacyclic wrap. Planned
            // scratch covers two matrices and all reused execution phases.
            unsafe {
                Self::truncated_product(
                    [left, right],
                    [dst],
                    [active_left_chunks, active_right_chunks],
                    [support, support],
                    [support],
                    &plan,
                    executor,
                    scratch,
                    |[left_prefix, right_prefix], work| {
                        SsaPointwise::pointwise_multiply_with_executor(
                            left_prefix,
                            right_prefix,
                            support,
                            leaf_budget,
                            plan.pointwise(),
                            executor,
                            work,
                        );
                    },
                );
            }
            return;
        }
        let half_len = plan.transform_len >> 1;
        let left_upper_half_zero = active_left_chunks <= half_len;
        let right_upper_half_zero = active_right_chunks <= half_len;

        // Every transform phase reuses the arena sized by this plan and executor.
        let parallelism = executor.parallelism().get();
        let slots = plan.parallel_slots(parallelism);
        // SAFETY: slots <= transform_len and the plan checked
        // mat_limbs = transform_len*inner_cl, so this subspan fits usize.
        let twiddle_len = unsafe { plan.inner_cl.get().unchecked_mul(slots) };
        debug_assert!(
            scratch.len() >= plan.transform_mul_scratch(parallelism),
            "SSA transform scratch is undersized: mod {}, inner {}, len {}, have {}, need {}, slots {}",
            plan.modulus_bits,
            plan.inner_bits,
            plan.transform_len,
            scratch.len(),
            plan.transform_mul_scratch(parallelism),
            slots
        );

        let dense =
            active_left_chunks == plan.transform_len && active_right_chunks == plan.transform_len;
        if parallelism == 1 && dense {
            // SAFETY: both ordinary inputs use all chunks, the sequential plan
            // sizes scratch, and dst holds the guarded or proven exact product.
            unsafe {
                DenseWorkspace::mul(left, right, &plan, scratch).reconstruct(dst);
            }
            return;
        }

        // Forward twiddles and pointwise work reuse the same tail. Once the
        // product consumes the right matrix, its storage joins the inverse and
        // reconstruction arena. Synchronous phase boundaries end every borrow.
        // SAFETY: transform_mul_scratch reserves both complete matrices.
        let (left_matrix, after_left) =
            unsafe { scratch.split_at_mut_unchecked(plan.mat_limbs.get()) };
        // SAFETY: after_left still contains the complete second matrix and workspace.
        let (right_matrix, after_right) =
            unsafe { after_left.split_at_mut_unchecked(plan.mat_limbs.get()) };

        // Stage each operand and run its forward FFT as one unit. Each branch
        // owns a disjoint matrix and twiddle arena; half-width active inputs fuse
        // their pre-twist with the first DIF stage during splitting.
        // SAFETY: the forward phase reserves two private twiddle arenas after
        // the matrices, even when they are consumed sequentially.
        let (left_twiddle, right_twiddle) = unsafe {
            let (left_work, after_left_work) = after_right.split_at_mut_unchecked(twiddle_len);
            let (right_work, _) = after_left_work.split_at_mut_unchecked(twiddle_len);
            (left_work, right_work)
        };
        if parallelism > 1 {
            let ((), ()) = executor.join(
                || {
                    // SAFETY: the left matrix and first twiddle span are disjoint,
                    // complete plan-sized partitions owned by this closure.
                    unsafe {
                        Self::stage_and_run_forward_fft(
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
                        Self::stage_and_run_forward_fft(
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
                Self::stage_and_run_forward_fft(
                    left,
                    left_matrix,
                    left_twiddle,
                    &plan,
                    left_upper_half_zero,
                    active_left_chunks,
                    executor,
                );
                Self::stage_and_run_forward_fft(
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

        // SAFETY: the matrices are perfectly sized for the complete transform.
        unsafe {
            SsaPointwise::pointwise_multiply_with_executor(
                left_matrix,
                right_matrix,
                plan.transform_len,
                leaf_budget,
                plan.pointwise(),
                executor,
                after_right,
            );
        }

        let active_out_chunks =
            significant_bits.map_or(plan.transform_len, |(left_bits, right_bits)| {
                left_bits
                    .saturating_add(right_bits)
                    .div_ceil(plan.chunk_bits.get())
                    .min(plan.transform_len)
            });

        // SAFETY: the consumed right matrix rejoins the workspace. The plan's
        // reconstruction phase reserves a twiddle prefix and accumulation tail.
        let (inverse_twiddle, recon_scratch) =
            unsafe { after_left.split_at_mut_unchecked(twiddle_len) };
        // SAFETY: matrices correctly sized, and the left twiddle arena is
        // disjoint and sized for the transform recursion.
        unsafe {
            Self::fft_in_place_with_executor(
                left_matrix,
                plan.transform_len,
                plan.twist_step_half,
                plan.inner_bits,
                true,
                active_out_chunks,
                executor,
                inverse_twiddle,
            );
        }

        // The inverse twiddle runs as its own sweep before accumulation, with the
        // same number of coefficient touches, but each coefficient's shift is
        // independent, so a parallel executor can fork the sweep.
        // SAFETY: the plan establishes the sign bound, complete inverse matrix,
        // disjoint reconstruction arena, and guarded or exact-product output.
        // Reconstruction folds to a canonical residue, so no post-pass
        // normalization is needed on either guard convention.
        unsafe {
            SsaCoefficients::reconstruct(
                // The polynomial degree is at most ceil(a/M)+ceil(b/M)-2,
                // below ceil((a+b)/M). Omitted outputs are therefore zero;
                // the partial inverse establishes exactly this complete prefix.
                left_matrix
                    .get_unchecked_mut(..active_out_chunks.unchecked_mul(plan.inner_cl.get())),
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                modulus_bits,
                dst,
                recon_scratch,
                Some((plan.inverse_twist(), inverse_twiddle)),
                executor,
            );
        }
    }
}
