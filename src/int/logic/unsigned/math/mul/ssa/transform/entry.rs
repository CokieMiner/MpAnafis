//! Public FFT transform entry points and top-level staging.

#![allow(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use crate::parallel::ParallelExecutor;

use super::{ArchKernels, Limb, SSA_PARALLEL_MIN_LIMB_WORK, SsaRing, SsaTransform};

/// Namespace for the Fermat-ring FFT: the butterflies, the matrix addressing
/// they run over, and the transforms that drive them end to end.
///
impl SsaTransform {
    /// Perform an in-place radix-2 FFT over the Fermat ring $\mathbb{Z}/(2^n + 1)$.
    ///
    /// `transform_len` coefficients of `cl = SsaRing::coeff_limbs(mod_bits)`
    /// limbs each are stored contiguously in `matrix`. The forward transform
    /// uses decimation-in-frequency and emits bit-reversed frequencies; the
    /// inverse uses decimation-in-time and consumes that order, so neither
    /// direction needs a coefficient permutation.
    ///
    /// # Arguments
    /// - `matrix`: flat coefficient buffer holding `transform_len * cl` limbs.
    /// - `transform_len`: number of coefficients, a power of two.
    /// - `root_shift`: twiddle factor $\omega = 2^{\text{root\_shift}}$ as a power-of-two shift.
    /// - `mod_bits`: Fermat modulus bit width $n$.
    /// - `inverse`: select the inverse DIT transform rather than forward DIF.
    /// - `active_chunks`: active coefficient bound ($L \le \text{transform\_len}$).
    /// - `executor`: parallel execution engine.
    /// - `scratch`: staging buffer of at least `cl` limbs for twiddle staging.
    ///
    /// # Safety
    /// `matrix` must cover `transform_len * coeff_limbs(mod_bits)` limbs.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "FFT dispatch needs the transform geometry, direction, executor, and scratch"
    )]
    pub unsafe fn fft_in_place_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        inverse: bool,
        active_chunks: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if transform_len < 2 || active_chunks == 0 {
            return;
        }
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();

        if !inverse {
            let half_len = transform_len >> 1;
            if active_chunks <= half_len {
                let half_matrix_len = half_len.wrapping_mul(cl);
                // SAFETY: active_chunks <= half_len, so upper half is initial zero.
                let (low_matrix, high_matrix) =
                    unsafe { matrix.split_at_mut_unchecked(half_matrix_len) };
                // Parallelize the sparse low→high copy when the active prefix is wide
                // enough to amortize fork/join and the executor has parallelism.
                if Self::has_parallel_work(active_chunks, cl, executor) {
                    let mid = active_chunks.wrapping_div(2);
                    let mid_twiddle =
                        SsaRing::reduce_mod_period(root_shift.wrapping_mul(mid), period);
                    let mid_offset = mid.wrapping_mul(cl);
                    let (low_left, low_right) = low_matrix.split_at_mut(mid_offset);
                    let (high_left, high_right) = high_matrix.split_at_mut(mid_offset);
                    let ((), ()) = executor.join(
                        || {
                            // SAFETY: both left slices contain `mid` complete,
                            // pairwise-disjoint coefficients.
                            unsafe {
                                scatter_twiddle_range(
                                    low_left, high_left, mid, 0, root_shift, mod_bits,
                                );
                            }
                        },
                        || {
                            let right_active = active_chunks.wrapping_sub(mid);
                            // SAFETY: both right slices contain `right_active`
                            // complete, pairwise-disjoint coefficients.
                            unsafe {
                                scatter_twiddle_range(
                                    low_right,
                                    high_right,
                                    right_active,
                                    mid_twiddle,
                                    root_shift,
                                    mod_bits,
                                );
                            }
                        },
                    );
                } else {
                    // SAFETY: both halves contain `active_chunks` complete,
                    // pairwise-disjoint coefficients.
                    unsafe {
                        scatter_twiddle_range(
                            low_matrix,
                            high_matrix,
                            active_chunks,
                            0,
                            root_shift,
                            mod_bits,
                        );
                    }
                }
                let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
                if Self::should_parallelize(half_len, cl, scratch.len(), executor) {
                    // SAFETY: low_matrix and high_matrix are disjoint and scratch is partitioned.
                    unsafe {
                        Self::recurse_dif_pair(
                            low_matrix,
                            high_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            active_chunks,
                            active_chunks,
                            executor,
                        );
                    }
                } else {
                    // SAFETY: both matrix halves and scratch are valid.
                    unsafe {
                        Self::fft_recursive_dif_with_executor(
                            low_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            active_chunks,
                            executor,
                        );
                        Self::fft_recursive_dif_with_executor(
                            high_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            active_chunks,
                            executor,
                        );
                    }
                }
            } else {
                // SAFETY: matrix and scratch are complete for transform_len.
                unsafe {
                    Self::fft_recursive_dif_with_executor(
                        matrix,
                        transform_len,
                        SsaRing::reduce_mod_period(root_shift, period),
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        active_chunks,
                        executor,
                    );
                }
            }
            return;
        }

        // Inverse DIT
        // SAFETY: matrix and scratch are complete for transform_len.
        unsafe {
            Self::fft_recursive_dit_with_executor(
                matrix,
                transform_len,
                SsaRing::reduce_mod_period(root_shift, period),
                mod_bits,
                scratch,
                add_sub_kernel,
                active_chunks,
                executor,
            );
        }
    }

    /// Complete a fused-stage DIF transform using an explicit executor.
    ///
    /// # Safety
    /// Preconditions identical to [`fft_in_place_with_executor`].
    pub unsafe fn fft_in_place_from_stage2_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        active_chunks: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if transform_len < 2 || active_chunks == 0 {
            return;
        }
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();
        let half_len = transform_len >> 1;
        let half_matrix_len = half_len.wrapping_mul(cl);
        // SAFETY: the validated matrix contains exactly two halves of
        // `half_len` complete coefficients.
        let (low_matrix, high_matrix) = unsafe { matrix.split_at_mut_unchecked(half_matrix_len) };
        let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
        // SAFETY: both halves are complete `half_len * cl` coefficient spans
        // and `scratch` still holds at least `cl` limbs.
        if Self::should_parallelize(half_len, cl, scratch.len(), executor) {
            // SAFETY: the helper partitions scratch into private slots; the two
            // matrix halves are disjoint.
            unsafe {
                Self::recurse_dif_pair(
                    low_matrix,
                    high_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    active_chunks,
                    active_chunks,
                    executor,
                );
            }
        } else {
            // SAFETY: both halves are complete and the sequential path reuses
            // scratch only after each child returns.
            unsafe {
                Self::fft_recursive_dif_with_executor(
                    low_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    active_chunks,
                    executor,
                );
                Self::fft_recursive_dif_with_executor(
                    high_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    active_chunks,
                    executor,
                );
            }
        }
    }

    /// Centralized grain and scratch policy for recursive transform splitting.
    pub fn should_parallelize<E: ParallelExecutor>(
        transform_len: usize,
        coeff_limbs: usize,
        scratch_len: usize,
        executor: &E,
    ) -> bool {
        let Some(two_slots) = coeff_limbs.checked_mul(2) else {
            return false;
        };
        Self::has_parallel_work(transform_len, coeff_limbs, executor) && scratch_len >= two_slots
    }

    /// Decides whether a disjoint range contains enough estimated limb work per
    /// Rayon worker to repay a fork. The generated threshold replaces the old
    /// fixed coefficient-count gates, which treated a 2-limb and a 1,000-limb
    /// coefficient as equal work.
    pub fn has_parallel_work<E: ParallelExecutor>(
        item_count: usize,
        unit_limb_work: usize,
        executor: &E,
    ) -> bool {
        let workers = executor.parallelism().get();
        if workers <= 1 || item_count < 2 {
            return false;
        }
        item_count
            .checked_mul(unit_limb_work)
            .is_some_and(|work| work.div_ceil(workers) >= SSA_PARALLEL_MIN_LIMB_WORK)
    }

    /// A radix-4 level can expose two independent child pairs when four private
    /// coefficient slots are available. This is a structural scratch condition,
    /// not a machine-specific grain threshold.
    pub fn can_fork_four(coeff_limbs: usize, scratch_len: usize) -> bool {
        coeff_limbs
            .checked_mul(4)
            .is_some_and(|four_slots| scratch_len >= four_slots)
    }
}

/// Copies and twists one active coefficient range into its disjoint high half.
///
/// # Safety
/// `low_matrix` and `high_matrix` must each contain at least `count` complete
/// `coeff_limbs(mod_bits)`-limb coefficients. Their active spans must not alias.
#[inline]
unsafe fn scatter_twiddle_range(
    low_matrix: &mut [Limb],
    high_matrix: &mut [Limb],
    count: usize,
    start_twiddle: usize,
    root_shift: usize,
    mod_bits: usize,
) {
    let cl = SsaRing::coeff_limbs(mod_bits);
    let period = mod_bits.wrapping_mul(2);
    let mut twiddle_shift = SsaRing::reduce_mod_period(start_twiddle, period);
    let twiddle_step = SsaRing::reduce_mod_period(root_shift, period);

    for i in 0..count {
        let offset = i.wrapping_mul(cl);
        // SAFETY: the caller proves both matrices contain `count` complete
        // coefficients, so this exact cl-limb slot is in bounds.
        let low_slot = unsafe { low_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
        // SAFETY: the same bound applies to the disjoint high matrix.
        let high_slot = unsafe { high_matrix.get_unchecked_mut(offset..offset.wrapping_add(cl)) };
        if twiddle_shift == 0 {
            high_slot.copy_from_slice(low_slot);
        } else {
            // SAFETY: the caller proves the active matrix spans do not alias;
            // both slots contain exactly cl limbs.
            unsafe {
                SsaRing::shift_from(high_slot, low_slot, twiddle_shift, mod_bits);
            }
        }
        twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
        if twiddle_shift >= period {
            twiddle_shift = twiddle_shift.wrapping_sub(period);
        }
    }
}
