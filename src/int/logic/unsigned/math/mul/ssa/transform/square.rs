//! Complete and truncated square-transform execution with reusable scratch.

#![expect(
    unsafe_code,
    reason = "Validated square plans establish coefficient supports and complete disjoint transform arenas"
)]

use crate::parallel::ParallelExecutor;

use super::{
    DenseWorkspace, Limb, SSA_BASE_MODULUS_BITS, SquareTransformInput, SsaCoefficients, SsaPlan,
    SsaPointwise, SsaRing, SsaTransform,
};

impl SsaTransform {
    /// Core recursive FFT squaring using the supplied synchronous executor.
    ///
    /// The pointwise squares fork over disjoint coefficient ranges, each with
    /// its own arena; nested coefficient squares run sequentially under a
    /// child executor so they cannot oversubscribe the outer one. The forward
    /// and inverse transform sweeps use the supplied executor and their own
    /// disjoint twiddle arena.
    ///
    /// # Safety
    /// `a` is a canonical guarded coefficient or an ordinary data prefix with
    /// implicit zero high limbs. `dst` contains a complete coefficient, except
    /// that a forced transform of an ordinary input may use a shorter output
    /// when it holds the exact square and that square is below `2^modulus_bits`.
    /// The basecase requires complete guarded input and output coefficients.
    /// `scratch` is sized from the selected plan and executor parallelism.
    /// When supplied, `forced_plan` was built for this exact modulus width and
    /// its executor-sized scratch geometry.
    #[expect(
        clippy::too_many_lines,
        reason = "FFT squaring orchestration linear pass"
    )]
    pub unsafe fn fft_sqr_mod_slices_with_executor<'plan, E: ParallelExecutor>(
        dst: &mut [Limb],
        a: &[Limb],
        modulus_bits: usize,
        force_transform: bool,
        forced_plan: impl Into<SquareTransformInput<'plan>>,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let ml = SsaRing::mod_limbs(modulus_bits);
        let cl = SsaRing::coeff_limbs(modulus_bits).get();
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
            // SAFETY: the unforced basecase contract supplies complete canonical
            // guarded input and output. The special-residue branch above removed
            // zero and -1; scratch contains the caller-sized basecase arena.
            unsafe {
                SsaPointwise::fermat_basecase_sqr_into(dst, a, modulus_bits, scratch);
            }
            return;
        }

        // Unforced squaring falls back to the square-optimal geometry.
        let plan = forced_plan.into().resolve(modulus_bits);
        let parallelism = executor.parallelism().get();
        let leaf_budget = plan.pointwise_leaf_count(parallelism);
        let slots = plan.parallel_slots(parallelism);
        // SAFETY: slots <= transform_len and the plan checked
        // mat_limbs = transform_len*inner_cl, so the twiddle subspan fits.
        let twiddle_len = unsafe { plan.inner_cl.get().unchecked_mul(slots) };
        debug_assert!(
            scratch.len() >= plan.transform_sqr_scratch(parallelism),
            "SSA square scratch is undersized for the executor's twiddle slots"
        );

        // The guard limb carries no chunk data, so the active chunk count comes
        // from the data width alone.
        let significant_width = if guarded {
            modulus_bits
        } else {
            // SAFETY: an unguarded operand has a.len() <= ml and the validated
            // modulus width is representable, so its capacity in bits fits.
            unsafe { SsaPlan::significant_bits_of_slice(a) }
        };
        let active_chunks = significant_width
            .div_ceil(plan.chunk_bits.get())
            .min(plan.transform_len);
        let support = Self::product_support(active_chunks, active_chunks, plan.transform_len);
        if support < plan.transform_len {
            // SAFETY: the input width proves support for the exact polynomial
            // square; the ITFT's omitted time tail is zero. The square plan
            // supplies the complete matrix and reused square/reconstruction arena.
            unsafe {
                Self::truncated_product(
                    [a],
                    [dst],
                    [active_chunks],
                    [support],
                    [support],
                    &plan,
                    executor,
                    scratch,
                    |[prefix], work| {
                        SsaPointwise::pointwise_square_with_executor(
                            prefix,
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
        let dense = active_chunks == plan.transform_len;
        if parallelism == 1 && dense {
            // SAFETY: the ordinary input uses all chunks, sequential scratch
            // covers the plan, and dst holds the guarded or proven exact square.
            unsafe {
                DenseWorkspace::sqr(a, &plan, scratch).reconstruct(dst);
            }
            return;
        }

        // The initialized matrix and staging arena are disjoint. Each staging
        // consumer overwrites its span before reading it, so no clearing is needed.
        // SAFETY: transform_sqr_scratch reserves this complete matrix first.
        let (matrix, after_matrix) =
            unsafe { scratch.split_at_mut_unchecked(plan.mat_limbs.get()) };
        let a_upper_half_zero = active_chunks <= (plan.transform_len >> 1);
        {
            // SAFETY: the square plan reserves this twiddle prefix after its matrix.
            let (twiddle_scratch, _) = unsafe { after_matrix.split_at_mut_unchecked(twiddle_len) };
            // A whole-bit twist and stage 1 DIF butterfly can be applied while splitting,
            // so the matrix is written once directly with stage 1 completed.
            let fused_stage1 = if a_upper_half_zero {
                // SAFETY: matrix is partitioned with plan.mat_limbs and twiddle_scratch
                // has the plan's executor-specific twiddle arena.
                unsafe {
                    SsaCoefficients::split_twisted_and_stage1_dif_with_executor(
                        a,
                        matrix,
                        plan.transform_len,
                        plan.chunk_bits,
                        plan.inner_cl,
                        plan.periods,
                        plan.twist_step_half,
                        plan.twist_step_half,
                        executor,
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
                    SsaCoefficients::split_twisted_with_executor(
                        a,
                        matrix,
                        plan.transform_len,
                        plan.chunk_bits,
                        plan.inner_cl,
                        plan.periods,
                        plan.twist_step_half,
                        executor,
                        twiddle_scratch,
                    );
                }
            }

            // SAFETY: all matrix and scratch spans were partitioned from the plan.
            // The fused flag selects the stage-2 entry when splitting already
            // completed the first DIF level; otherwise the full forward runs.
            unsafe {
                if fused_stage1 {
                    Self::fft_in_place_from_stage2_with_executor(
                        matrix,
                        plan.transform_len,
                        plan.twist_step_half,
                        plan.inner_bits,
                        active_chunks,
                        executor,
                        twiddle_scratch,
                    );
                } else {
                    Self::fft_in_place_with_executor(
                        matrix,
                        plan.transform_len,
                        plan.twist_step_half,
                        plan.inner_bits,
                        false,
                        active_chunks,
                        executor,
                        twiddle_scratch,
                    );
                }
                SsaPointwise::pointwise_square_with_executor(
                    matrix,
                    plan.transform_len,
                    leaf_budget,
                    plan.pointwise(),
                    executor,
                    after_matrix,
                );
            }
        }
        // SAFETY: the pointwise phase has ended; the plan reserves this prefix
        // and reconstruction tail for the inverse phase.
        let (inverse_twiddle, recon_scratch) =
            unsafe { after_matrix.split_at_mut_unchecked(twiddle_len) };
        // SAFETY: the inverse reuses the complete matrix and disjoint twiddle
        // arena after all pointwise square borrows have ended.
        let active_out_chunks = significant_width
            .saturating_mul(2)
            .div_ceil(plan.chunk_bits.get())
            .min(plan.transform_len);
        // SAFETY: the matrix and twiddle partitions are complete and disjoint.
        unsafe {
            Self::fft_in_place_with_executor(
                matrix,
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
        // SAFETY: the square plan establishes the coefficient sign bound,
        // complete inverse matrix, disjoint arena, and exact output width.
        unsafe {
            SsaCoefficients::reconstruct(
                // The significant-width bound proves all omitted square
                // coefficients zero; only this prefix is established by DIT.
                matrix.get_unchecked_mut(..active_out_chunks.unchecked_mul(plan.inner_cl.get())),
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
        // Reconstruction folds to a canonical residue. Exact full-width
        // squares may omit the guard because their significant-bit bound
        // proves the result is below 2^modulus_bits.
    }
}
