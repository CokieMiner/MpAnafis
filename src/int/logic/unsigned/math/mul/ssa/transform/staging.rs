//! Operand staging and the first forward DIF stage.

#![expect(
    unsafe_code,
    reason = "Admitted transform plans establish staging spans, zero tails, and scratch coefficients"
)]

use crate::parallel::ParallelExecutor;

use super::{FftPlan, Limb, SsaCoefficients, SsaTransform};

impl SsaTransform {
    /// Stages an operand and runs its forward transform.
    ///
    /// When the active chunks fit the lower half, staging combines the
    /// pre-twist with the first DIF level and execution resumes at stage two.
    ///
    /// # Safety
    /// `matrix` and `twiddle` are disjoint complete initialized plan partitions.
    /// `active_chunks` covers `src` under the plan, and `upper_half_zero` may
    /// be true only when those chunks fit in the lower transform half.
    pub unsafe fn stage_and_run_forward_fft<E: ParallelExecutor>(
        src: &[Limb],
        matrix: &mut [Limb],
        twiddle: &mut [Limb],
        plan: &FftPlan,
        upper_half_zero: bool,
        active_chunks: usize,
        executor: &E,
    ) {
        let fused_stage1 = if upper_half_zero {
            // SAFETY: the complete plan partitions are disjoint and the
            // admitted support fits the lower half for first-stage fusion.
            unsafe {
                SsaCoefficients::split_twisted_and_stage1_dif_with_executor(
                    src,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    plan.twist_step_half,
                    executor,
                    twiddle,
                )
            }
        } else {
            false
        };
        if !fused_stage1 {
            // SAFETY: src fits the complete matrix under the plan's chunk
            // partition; staging initializes its coefficients and zero tail.
            unsafe {
                SsaCoefficients::split_twisted_with_executor(
                    src,
                    matrix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    executor,
                    twiddle,
                );
            }
        }
        // SAFETY: staging produced complete coefficients. The fusion flag
        // identifies whether the first DIF level has already been executed.
        unsafe {
            if fused_stage1 {
                Self::fft_in_place_from_stage2_with_executor(
                    matrix,
                    plan.transform_len,
                    plan.twist_step_half,
                    plan.inner_bits,
                    active_chunks,
                    executor,
                    twiddle,
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
                    twiddle,
                );
            }
        }
    }
}
