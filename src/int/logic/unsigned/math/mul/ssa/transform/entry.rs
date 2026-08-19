//! Public FFT transform entry points and top-level staging.

#![allow(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use crate::parallel::ParallelExecutor;

use super::{
    ArchKernels, Limb, SsaRing, SsaTransform, fft_recursive_dif_with_executor,
    fft_recursive_dit_with_executor, recurse_dif_pair,
};

/// Namespace for the Fermat-ring FFT: the butterflies, the matrix addressing
/// they run over, and the transforms that drive them end to end.
///
impl SsaTransform {
    #[allow(
        clippy::too_many_lines,
        reason = "the paired DIF/DIT loops share one proved buffer layout; splitting them would add one-use wrappers and hide the transform invariant"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "Internal algorithm needs all buffers passed explicitly"
    )]
    /// Perform an in-place radix-2 FFT over the Fermat ring
    /// $\mathbb{Z}/(2^n + 1)$.
    ///
    /// `transform_len` coefficients of `cl = SsaRing::coeff_limbs(mod_bits)`
    /// limbs each are stored contiguously in `matrix`. The forward transform
    /// uses decimation-in-frequency and emits bit-reversed frequencies; the
    /// inverse uses decimation-in-time and consumes that order, so neither
    /// direction needs a coefficient permutation.
    ///
    /// # Arguments
    ///
    /// - `matrix`: flat coefficient buffer holding `transform_len * cl` limbs.
    /// - `transform_len`: number of coefficients, a power of two.
    /// - `root_shift`: twiddle factor $\omega = 2^{\text{root\_shift}}$ as a
    ///   power-of-two shift.
    /// - `mod_bits`: Fermat modulus bit width $n$.
    /// - `inverse`: select the inverse DIT transform rather than forward DIF.
    /// - `upper_half_zero`: forward-only; the specialized first DIF stage
    ///   requires the upper coefficient half of `matrix` to be zero.
    /// - `scratch`: staging buffer of at least `cl` limbs for twiddle staging.
    ///
    /// Perform an in-place FFT while recursively scheduling independent
    /// coefficient ranges through `executor`.
    ///
    /// Parallel recursion is enabled only when the transform has enough
    /// coefficient work and the caller supplied two disjoint twiddle slots.
    /// This keeps the cache-contiguous matrix layout and never shares mutable
    /// scratch between concurrently running branches.
    ///
    /// # Safety
    /// The matrix, root, and scratch preconditions are identical to
    /// the matrix, root, and scratch preconditions documented below. In
    /// addition, `executor` must uphold the
    /// [`ParallelExecutor`] join contract.
    #[allow(
        clippy::too_many_arguments,
        reason = "FFT dispatch needs the transform geometry, direction, executor, and scratch"
    )]
    pub unsafe fn fft_in_place_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        inverse: bool,
        upper_half_zero: bool,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();

        if !inverse {
            if upper_half_zero {
                let half_len = transform_len >> 1;
                let half_matrix_len = half_len.wrapping_mul(cl);
                // SAFETY: the validated matrix contains exactly two halves of
                // `half_len` complete coefficients.
                let (low_matrix, high_matrix) =
                    unsafe { matrix.split_at_mut_unchecked(half_matrix_len) };
                debug_assert!(
                    high_matrix.iter().all(|limb| *limb == 0),
                    "the specialized DIF stage requires a zero upper matrix half"
                );
                let mut twiddle_shift = 0_usize;
                for (low_slot, high_slot) in low_matrix
                    .chunks_exact_mut(cl)
                    .zip(high_matrix.chunks_exact_mut(cl))
                {
                    // With high=0, the first DIF butterfly is exactly
                    // (low+high, (low-high)*w) = (low, low*w).
                    if twiddle_shift == 0 {
                        high_slot.copy_from_slice(low_slot);
                    } else {
                        // SAFETY: low_slot and high_slot are disjoint cl-limb
                        // spans and the low coefficient is canonical.
                        unsafe {
                            SsaRing::shift_from(high_slot, low_slot, twiddle_shift, mod_bits);
                        }
                    }
                    twiddle_shift = twiddle_shift.wrapping_add(root_shift);
                    if twiddle_shift >= period {
                        twiddle_shift = twiddle_shift.wrapping_sub(period);
                    }
                }
                // The specialized stage consumed the outermost DIF level, so the two
                // contiguous halves are now independent transforms at twice the root.
                let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(2), period);
                // SAFETY: both halves are complete `half_len * cl` coefficient spans
                // and `scratch` still holds at least `cl` limbs.
                if should_parallelize(half_len, cl, scratch.len(), executor) {
                    // SAFETY: the helper partitions scratch into private slots;
                    // the two matrix halves are disjoint.
                    unsafe {
                        recurse_dif_pair(
                            low_matrix,
                            high_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            executor,
                        );
                    }
                } else {
                    // SAFETY: both halves are complete and the sequential path
                    // reuses scratch only after each child returns.
                    unsafe {
                        fft_recursive_dif_with_executor(
                            low_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            executor,
                        );
                        fft_recursive_dif_with_executor(
                            high_matrix,
                            half_len,
                            next_root,
                            mod_bits,
                            scratch,
                            add_sub_kernel,
                            executor,
                        );
                    }
                }
            } else {
                // SAFETY: the caller guarantees `transform_len * cl` limbs and a
                // `cl`-limb scratch slot.
                unsafe {
                    fft_recursive_dif_with_executor(
                        matrix,
                        transform_len,
                        SsaRing::reduce_mod_period(root_shift, period),
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        executor,
                    );
                }
            }
            return;
        }

        // SAFETY: the caller guarantees `transform_len * cl` limbs and a `cl`-limb
        // scratch slot.
        unsafe {
            fft_recursive_dit_with_executor(
                matrix,
                transform_len,
                SsaRing::reduce_mod_period(root_shift, period),
                mod_bits,
                scratch,
                add_sub_kernel,
                executor,
            );
        }
    }

    /// Complete a fused-stage DIF transform using an explicit executor.
    ///
    /// # Safety
    /// The matrix and scratch preconditions are those of the fused-stage DIF
    /// transform described by the caller.
    pub unsafe fn fft_in_place_from_stage2_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if transform_len < 2 {
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
        if should_parallelize(half_len, cl, scratch.len(), executor) {
            // SAFETY: the helper partitions scratch into private slots; the two
            // matrix halves are disjoint.
            unsafe {
                recurse_dif_pair(
                    low_matrix,
                    high_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    executor,
                );
            }
        } else {
            // SAFETY: both halves are complete and the sequential path reuses
            // scratch only after each child returns.
            unsafe {
                fft_recursive_dif_with_executor(
                    low_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    executor,
                );
                fft_recursive_dif_with_executor(
                    high_matrix,
                    half_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    executor,
                );
            }
        }
    }
}

/// Minimum transform width for recursive fork/join. Smaller transforms stay
/// contiguous and sequential because scheduling overhead dominates their work.
const MIN_PARALLEL_TRANSFORM_LEN: usize = 8;

/// Centralized grain and scratch policy for recursive transform splitting.
pub fn should_parallelize<E: ParallelExecutor>(
    transform_len: usize,
    coeff_limbs: usize,
    scratch_len: usize,
    executor: &E,
) -> bool {
    let workers = executor.parallelism().get();
    let Some(two_slots) = coeff_limbs.checked_mul(2) else {
        return false;
    };
    transform_len >= MIN_PARALLEL_TRANSFORM_LEN
        && transform_len >= workers.saturating_mul(2)
        && scratch_len >= two_slots
}

/// A radix-4 level can expose two independent child pairs when four private
/// coefficient slots are available. This is a structural scratch condition,
/// not a machine-specific grain threshold.
pub fn can_fork_four(coeff_limbs: usize, scratch_len: usize) -> bool {
    coeff_limbs
        .checked_mul(4)
        .is_some_and(|four_slots| scratch_len >= four_slots)
}
