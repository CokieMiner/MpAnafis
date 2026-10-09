//! Shared-operand transform orchestration.

#![expect(
    unsafe_code,
    reason = "Validated paired-product layouts partition disjoint matrices and complete per-worker arenas"
)]

use crate::parallel::ParallelExecutor;

use super::{
    Limb, MulTransformInput, SSA_BASE_MODULUS_BITS, SsaCoefficients, SsaPointwise, SsaRing,
    SsaTransform,
};

impl SsaTransform {
    /// Multiply `a` and `b` by the shared coefficient `x` in one Fermat ring.
    ///
    /// The operation transforms `x` once, then performs two pointwise products
    /// and two independent inverse transforms.
    ///
    /// # Safety
    /// The operands and destinations follow one consistent guard convention.
    /// `forced_plan`, when present, was built for `modulus_bits`, and `scratch`
    /// has its complete executor-sized two-by-one layout.
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::similar_names,
        reason = "FFT two-by-one orchestration linear pass"
    )]
    pub unsafe fn fft_mul_two_by_one_mod_slices_with_executor<'plan, E: ParallelExecutor>(
        dst_a: &mut [Limb],
        dst_b: &mut [Limb],
        a: &[Limb],
        b: &[Limb],
        x: &[Limb],
        modulus_bits: usize,
        significant_bits: Option<(usize, usize, usize)>,
        force_transform: bool,
        forced_plan: impl Into<MulTransformInput<'plan>>,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let output_has_guard = dst_a.len() > ml;
        debug_assert_eq!(
            output_has_guard,
            dst_b.len() > ml,
            "both destinations must match guard convention"
        );
        let a_guarded = a.len() > ml;
        let b_guarded = b.len() > ml;
        let x_guarded = x.len() > ml;
        debug_assert!(
            a_guarded == b_guarded && b_guarded == x_guarded,
            "SSA operands must all include the guard limb or all omit it"
        );
        let guarded_operands = a_guarded;
        if guarded_operands {
            // SAFETY: guarded operands and destinations have complete ring widths.
            let special_a =
                unsafe { SsaPointwise::write_special_residue_product(dst_a, a, x, modulus_bits) };
            // SAFETY: guarded operands and destinations have complete ring widths.
            let special_b =
                unsafe { SsaPointwise::write_special_residue_product(dst_b, b, x, modulus_bits) };
            if special_a && special_b {
                return;
            }
            if special_a {
                // SAFETY: the boundary contract also establishes one ordinary product.
                unsafe {
                    Self::fft_mul_mod_slices_with_executor(
                        dst_b,
                        b,
                        x,
                        modulus_bits,
                        None,
                        force_transform,
                        forced_plan,
                        executor,
                        scratch,
                    );
                }
                return;
            }
            if special_b {
                // SAFETY: the boundary contract also establishes one ordinary product.
                unsafe {
                    Self::fft_mul_mod_slices_with_executor(
                        dst_a,
                        a,
                        x,
                        modulus_bits,
                        None,
                        force_transform,
                        forced_plan,
                        executor,
                        scratch,
                    );
                }
                return;
            }
        }

        if modulus_bits <= SSA_BASE_MODULUS_BITS && !force_transform {
            debug_assert!(
                guarded_operands && output_has_guard,
                "implicit guards and exact outputs require the transform path"
            );
            // SAFETY: the guarded coefficients are canonical and non-special.
            // The two products reuse the complete caller-owned basecase arena.
            unsafe {
                SsaPointwise::fermat_basecase_mul_into(dst_a, a, x, modulus_bits, scratch);
                SsaPointwise::fermat_basecase_mul_into(dst_b, b, x, modulus_bits, scratch);
            }
            return;
        }

        // Unforced pairs fall back to the pair-optimal geometry.
        let plan = forced_plan.into().resolve(modulus_bits, true);
        let leaf_budget = plan.pointwise_leaf_count(executor.parallelism().get());
        let (active_a_chunks, active_b_chunks, active_x_chunks) = significant_bits.map_or(
            (plan.transform_len, plan.transform_len, plan.transform_len),
            |(sig_a, sig_b, sig_x)| {
                (
                    sig_a
                        .div_ceil(plan.chunk_bits.get())
                        .min(plan.transform_len),
                    sig_b
                        .div_ceil(plan.chunk_bits.get())
                        .min(plan.transform_len),
                    sig_x
                        .div_ceil(plan.chunk_bits.get())
                        .min(plan.transform_len),
                )
            },
        );
        let support_a = Self::product_support(active_a_chunks, active_x_chunks, plan.transform_len);
        let support_b = Self::product_support(active_b_chunks, active_x_chunks, plan.transform_len);
        if support_a < plan.transform_len || support_b < plan.transform_len {
            // SAFETY: each prefix bounds its output polynomial; a full prefix
            // permits ring wrap, while a shorter one proves an implicit zero
            // tail. The shared spectrum covers the larger of the two requests.
            unsafe {
                Self::truncated_product(
                    [a, b, x],
                    [dst_a, dst_b],
                    [active_a_chunks, active_b_chunks, active_x_chunks],
                    [support_a, support_b, support_a.max(support_b)],
                    [support_a, support_b],
                    &plan,
                    executor,
                    scratch,
                    |[a_prefix, b_prefix, x_prefix], work| {
                        let common = support_a.min(support_b);
                        // common <= K and K*inner_cl is the validated matrix span.
                        let span = common.unchecked_mul(plan.inner_cl.get());
                        let (a_common, a_tail) = a_prefix.split_at_mut_unchecked(span);
                        let (b_common, b_tail) = b_prefix.split_at_mut_unchecked(span);
                        let (x_common, x_tail) = x_prefix.split_at_mut_unchecked(span);
                        SsaPointwise::pointwise_multiply_pair_with_executor(
                            [a_common, b_common, x_common],
                            common,
                            leaf_budget,
                            plan.pointwise(),
                            executor,
                            work,
                        );
                        if support_a > common {
                            SsaPointwise::pointwise_multiply_with_executor(
                                a_tail,
                                x_tail,
                                support_a.unchecked_sub(common),
                                leaf_budget,
                                plan.pointwise(),
                                executor,
                                work,
                            );
                        } else if support_b > common {
                            SsaPointwise::pointwise_multiply_with_executor(
                                b_tail,
                                x_tail,
                                support_b.unchecked_sub(common),
                                leaf_budget,
                                plan.pointwise(),
                                executor,
                                work,
                            );
                        }
                    },
                );
            }
            return;
        }
        let half_len = plan.transform_len >> 1;
        let a_upper_half_zero = active_a_chunks <= half_len;
        let b_upper_half_zero = active_b_chunks <= half_len;
        let x_upper_half_zero = active_x_chunks <= half_len;

        let parallelism = executor.parallelism().get();
        let slots = plan.parallel_slots(parallelism);
        // The admitted layout bounds every complete twiddle partition.
        // SAFETY: slots<=K and K*inner_cl is the plan's checked matrix span.
        let twiddle_len = unsafe { plan.inner_cl.get().unchecked_mul(slots) };
        debug_assert!(
            scratch.len() >= plan.transform_mul_two_by_one_scratch(parallelism),
            "SSA two-by-one scratch is undersized"
        );

        // SAFETY: the pair plan checks three complete disjoint matrices
        // before its reusable twiddle/pointwise arena.
        let (a_matrix, b_matrix, after_b) = unsafe {
            let (a_matrix, after_a) = scratch.split_at_mut_unchecked(plan.mat_limbs.get());
            let (b_matrix, after_b) = after_a.split_at_mut_unchecked(plan.mat_limbs.get());
            (a_matrix, b_matrix, after_b)
        };
        // SAFETY: the pair plan reserves a third complete matrix and its arena.
        // This reborrow ends before the dead shared matrix joins reconstruction.
        let (x_matrix, after_x) = unsafe { after_b.split_at_mut_unchecked(plan.mat_limbs.get()) };

        // A dense sequential pair completes forwards, both shared products,
        // and both inverses one cache-resident subtree at a time.
        let dense = active_a_chunks == plan.transform_len
            && active_b_chunks == plan.transform_len
            && active_x_chunks == plan.transform_len;
        let fused = parallelism == 1 && dense;
        if fused {
            // SAFETY: the pair plan reserves at least two complete staging
            // coefficients beyond its three matrices.
            let (stage_span, _) =
                unsafe { after_x.split_at_mut_unchecked(plan.inner_cl.get().unchecked_mul(2)) };
            // SAFETY: dense operands produce complete coefficient matrices;
            // the staging span is a disjoint complete coefficient, reused for
            // each operand only after the previous staging completes.
            unsafe {
                SsaCoefficients::split_twisted(
                    x,
                    x_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    stage_span,
                );
                SsaCoefficients::split_twisted(
                    a,
                    a_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    stage_span,
                );
                SsaCoefficients::split_twisted(
                    b,
                    b_matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    stage_span,
                );
            }
            // SAFETY: the staged matrices are complete and the arena covers
            // the planned scratch for convolve_pair_subtrees.
            unsafe {
                Self::convolve_pair_subtrees(
                    [a_matrix, b_matrix, x_matrix],
                    plan.transform_len,
                    plan.twist_step_half,
                    plan.pointwise(),
                    executor,
                    after_x,
                );
            }
        } else {
            // SAFETY: the checked forward arena contains both complete private
            // twiddle partitions after all three matrices.
            let (twiddle_1, twiddle_2) = unsafe {
                let (twiddle_1, after_t1) = after_x.split_at_mut_unchecked(twiddle_len);
                let (twiddle_2, _) = after_t1.split_at_mut_unchecked(twiddle_len);
                (twiddle_1, twiddle_2)
            };
            // SAFETY: the matrix and twiddle partitions are complete and disjoint.
            unsafe {
                Self::stage_and_run_forward_fft(
                    x,
                    x_matrix,
                    twiddle_1,
                    &plan,
                    x_upper_half_zero,
                    active_x_chunks,
                    executor,
                );
            }

            if parallelism > 1 {
                let ((), ()) = executor.join(
                    || {
                        // SAFETY: this closure owns disjoint matrix and twiddle partitions.
                        unsafe {
                            Self::stage_and_run_forward_fft(
                                a,
                                a_matrix,
                                twiddle_1,
                                &plan,
                                a_upper_half_zero,
                                active_a_chunks,
                                executor,
                            );
                        }
                    },
                    || {
                        // SAFETY: this closure owns disjoint matrix and twiddle partitions.
                        unsafe {
                            Self::stage_and_run_forward_fft(
                                b,
                                b_matrix,
                                twiddle_2,
                                &plan,
                                b_upper_half_zero,
                                active_b_chunks,
                                executor,
                            );
                        }
                    },
                );
            } else {
                // SAFETY: sequential staging reuses one complete twiddle arena.
                unsafe {
                    Self::stage_and_run_forward_fft(
                        a,
                        a_matrix,
                        twiddle_1,
                        &plan,
                        a_upper_half_zero,
                        active_a_chunks,
                        executor,
                    );
                    Self::stage_and_run_forward_fft(
                        b,
                        b_matrix,
                        twiddle_1,
                        &plan,
                        b_upper_half_zero,
                        active_b_chunks,
                        executor,
                    );
                }
            }
        }

        // SAFETY: matrices and scratch are exact plan partitions.
        // Either fused mode already multiplied its subtrees above.
        if !fused {
            // SAFETY: the two pointwise products consume exact plan partitions.
            unsafe {
                SsaPointwise::pointwise_multiply_pair_with_executor(
                    [a_matrix, b_matrix, x_matrix],
                    plan.transform_len,
                    leaf_budget,
                    plan.pointwise(),
                    executor,
                    after_x,
                );
            }
        }

        // The shared matrix is dead after both products; its storage becomes
        // the inverse-twiddle and reconstruction workspace for both chains.
        let active_out_chunks_a =
            significant_bits.map_or(plan.transform_len, |(sig_a, _, sig_x)| {
                sig_a
                    .saturating_add(sig_x)
                    .div_ceil(plan.chunk_bits.get())
                    .min(plan.transform_len)
            });
        let active_out_chunks_b =
            significant_bits.map_or(plan.transform_len, |(_, sig_b, sig_x)| {
                sig_b
                    .saturating_add(sig_x)
                    .div_ceil(plan.chunk_bits.get())
                    .min(plan.transform_len)
            });

        // One output chain consumes its inverse-twiddle span and reconstruction
        // arena together. Sequential chains reuse the same storage; concurrent
        // chains receive disjoint complete partitions.
        let finish = |matrix: &mut [Limb], dst: &mut [Limb], active: usize, work: &mut [Limb]| {
            // SAFETY: each planned output chain reserves the complete twiddle
            // partition before its disjoint reconstruction arena.
            let (inverse_twiddle, recon) = unsafe { work.split_at_mut_unchecked(twiddle_len) };
            // SAFETY: matrix is a complete initialized plan partition; work
            // contains its disjoint twiddle and reconstruction spans. The pair
            // plan proves the centered coefficient bound and output width.
            // A fused traversal has already inverted every subtree.
            unsafe {
                if !fused {
                    Self::fft_in_place_with_executor(
                        matrix,
                        plan.transform_len,
                        plan.twist_step_half,
                        plan.inner_bits,
                        true,
                        active,
                        executor,
                        inverse_twiddle,
                    );
                }
                // active <= K and K*inner_cl is the validated matrix span.
                // The inverse initializes the complete requested prefix.
                SsaCoefficients::reconstruct(
                    matrix.get_unchecked_mut(..active.unchecked_mul(plan.inner_cl.get())),
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    modulus_bits,
                    dst,
                    recon,
                    Some((plan.inverse_twist(), inverse_twiddle)),
                    executor,
                );
            }
        };

        if parallelism > 1 {
            // The two inverse-and-reconstruct chains are independent, so a
            // parallel executor runs them concurrently. Each chain owns a
            // private twiddle arena and reconstruction workspace carved from
            // the dead shared matrix and forward-twiddle storage, sized by the
            // concurrent branch of `transform_mul_two_by_one_scratch`.
            // SAFETY: transform_mul_two_by_one_scratch checks both simultaneous
            // output chains, each including its twiddle and reconstruction arena.
            let (first, second) = unsafe {
                let chain_len = twiddle_len.unchecked_add(plan.reconstruction_scratch(parallelism));
                after_b.split_at_mut_unchecked(chain_len)
            };
            let ((), ()) = executor.join(
                || finish(a_matrix, dst_a, active_out_chunks_a, first),
                || finish(b_matrix, dst_b, active_out_chunks_b, second),
            );
        } else {
            finish(a_matrix, dst_a, active_out_chunks_a, after_b);
            finish(b_matrix, dst_b, active_out_chunks_b, after_b);
        }
    }
}
