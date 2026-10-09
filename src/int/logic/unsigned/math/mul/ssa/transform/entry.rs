//! FFT entry points and flat coefficient addressing.

#![expect(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{
    AddSubFromKernel, ArchKernels, Limb, SSA_PARALLEL_MIN_LIMB_WORK, SsaRing, SsaTransform,
};

/// Invariant ring dimensions, butterfly kernel, and executor for FFT workers.
/// Recursive stages borrow this context instead of recomputing slot widths,
/// periods, or CPU kernel selection.
#[derive(Clone, Copy)]
pub struct TransformContext<'exec, E> {
    /// Fermat modulus bit width n.
    pub mod_bits: usize,
    /// Limb count per coefficient, including its guard.
    pub cl: NonZeroUsize,
    /// Primitive root period 2n.
    pub period: NonZeroUsize,
    /// Simultaneous addition and subtraction kernel for butterflies.
    pub kernel: AddSubFromKernel,
    /// Parallel execution engine.
    pub executor: &'exec E,
}

impl<'exec, E: ParallelExecutor> TransformContext<'exec, E> {
    /// Constructs shared execution parameters for an admitted ring geometry.
    #[inline]
    pub const fn new(mod_bits: usize, kernel: AddSubFromKernel, executor: &'exec E) -> Self {
        Self {
            mod_bits,
            cl: SsaRing::coeff_limbs(mod_bits),
            // SAFETY: all callers construct this context from a ring geometry
            // that validated positive mod_bits and representable 4*mod_bits.
            period: unsafe { NonZeroUsize::new_unchecked(mod_bits.unchecked_mul(2)) },
            kernel,
            executor,
        }
    }
}

impl SsaTransform {
    /// Returns a shared slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[expect(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff(buf: &[Limb], index: usize, coeff_limbs: usize) -> &[Limb] {
        // SAFETY: index<transform_len and the complete transform_len*width
        // span belongs to buf, so both this offset and its end fit usize.
        unsafe {
            let offset = index.unchecked_mul(coeff_limbs);
            buf.get_unchecked(offset..offset.unchecked_add(coeff_limbs))
        }
    }

    /// Returns a mutable slice of the coefficient at `index`.
    ///
    /// # Safety
    /// `index < transform_len` and `buf.len() >= transform_len * coeff_limbs`.
    #[expect(
        clippy::inline_always,
        reason = "zero-cost pointer arithmetic on the hot FFT path"
    )]
    #[inline(always)]
    pub unsafe fn coeff_mut(buf: &mut [Limb], index: usize, coeff_limbs: usize) -> &mut [Limb] {
        // SAFETY: index<transform_len and the complete transform_len*width
        // span belongs exclusively to buf, so this offset and its end fit.
        unsafe {
            let offset = index.unchecked_mul(coeff_limbs);
            buf.get_unchecked_mut(offset..offset.unchecked_add(coeff_limbs))
        }
    }
    /// Transforms coefficients over the Fermat ring $\mathbb{Z}/(2^n + 1)$.
    ///
    /// `transform_len` coefficients of `cl = SsaRing::coeff_limbs(mod_bits)`
    /// limbs each are stored contiguously in `matrix`. The forward transform
    /// uses decimation-in-frequency and emits bit-reversed frequencies; the
    /// inverse uses decimation-in-time and consumes that order, so neither
    /// direction needs a coefficient permutation. Complete subtrees use
    /// radix-four stages with binary leaves.
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
    /// The positive power-of-two length divides the representable period
    /// `2*mod_bits`, and `root_shift*transform_len=2*mod_bits` is its principal
    /// exponent. The admitted ring has representable `4*mod_bits`. The initialized
    /// matrix covers `transform_len * coeff_limbs(mod_bits)` limbs, scratch
    /// contains at least one complete coefficient, and `active_chunks<=transform_len`.
    #[expect(
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
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();
        let ctx = TransformContext::new(mod_bits, add_sub_kernel, executor);
        let cl = ctx.cl.get();

        if !inverse {
            let half_len = transform_len >> 1;
            if active_chunks <= half_len {
                // SAFETY: half_len*cl is one half of the complete caller matrix.
                let half_matrix_len = unsafe { half_len.unchecked_mul(cl) };
                // SAFETY: active_chunks <= half_len, so upper half is initial zero.
                let (low_matrix, high_matrix) =
                    unsafe { matrix.split_at_mut_unchecked(half_matrix_len) };
                // Independent sparse copies use the executor's worker budget;
                // recursive forks partition it between sibling subtrees.
                if Self::has_parallel_work(active_chunks, cl, executor.parallelism().get()) {
                    let mid = active_chunks >> 1;
                    // SAFETY: mid<half_len; the principal exponent at this
                    // index is below period/2. Both matrices contain half_len
                    // complete initialized slots, so the aligned split fits.
                    let (mid_twiddle, (low_left, low_right), (high_left, high_right)) = unsafe {
                        let mid_offset = mid.unchecked_mul(cl);
                        (
                            root_shift.unchecked_mul(mid),
                            low_matrix.split_at_mut_unchecked(mid_offset),
                            high_matrix.split_at_mut_unchecked(mid_offset),
                        )
                    };
                    let ((), ()) = executor.join(
                        || {
                            // SAFETY: both left slices contain `mid` complete,
                            // pairwise-disjoint coefficients.
                            unsafe {
                                Self::scatter_twiddle_range(
                                    low_left, high_left, mid, 0, root_shift, mod_bits,
                                );
                            }
                        },
                        || {
                            // SAFETY: mid=floor(active_chunks/2)<=active_chunks.
                            let right_active = unsafe { active_chunks.unchecked_sub(mid) };
                            // SAFETY: both right slices contain `right_active`
                            // complete, pairwise-disjoint coefficients.
                            unsafe {
                                Self::scatter_twiddle_range(
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
                        Self::scatter_twiddle_range(
                            low_matrix,
                            high_matrix,
                            active_chunks,
                            0,
                            root_shift,
                            mod_bits,
                        );
                    }
                }
                // SAFETY: transform_len>=2 and root_shift*transform_len=period.
                // Doubling is the child's principal exponent and at most period.
                let next_root = unsafe { root_shift.unchecked_mul(2) };
                // Subtree work scales with its remaining binary depth and
                // coefficient width; that work determines fork admission.
                #[expect(
                    clippy::as_conversions,
                    reason = "the logarithm of a positive usize power of two is below usize::BITS"
                )]
                let child_levels = half_len.trailing_zeros() as usize;
                // SAFETY: log2(half_len)<=half_len, whose complete cl-limb
                // slots occupy half the representable matrix.
                let child_work = unsafe { cl.unchecked_mul(child_levels) };
                if Self::should_parallelize(
                    half_len,
                    child_work,
                    cl,
                    scratch.len(),
                    executor.parallelism().get(),
                ) {
                    // SAFETY: low_matrix and high_matrix are disjoint and scratch is partitioned.
                    unsafe {
                        Self::recurse_dif_pair(
                            low_matrix,
                            high_matrix,
                            half_len,
                            next_root,
                            scratch,
                            active_chunks,
                            active_chunks,
                            &ctx,
                        );
                    }
                } else {
                    // SAFETY: both matrix halves and scratch are valid.
                    unsafe {
                        Self::fft_recursive_dif_with_executor(
                            low_matrix,
                            half_len,
                            next_root,
                            scratch,
                            active_chunks,
                            &ctx,
                        );
                        Self::fft_recursive_dif_with_executor(
                            high_matrix,
                            half_len,
                            next_root,
                            scratch,
                            active_chunks,
                            &ctx,
                        );
                    }
                }
            } else {
                // SAFETY: matrix and scratch are complete for transform_len.
                unsafe {
                    Self::fft_recursive_dif_with_executor(
                        matrix,
                        transform_len,
                        root_shift,
                        scratch,
                        active_chunks,
                        &ctx,
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
                root_shift,
                scratch,
                active_chunks,
                &ctx,
            );
        }
    }

    /// Complete a fused-stage DIF transform using an explicit executor.
    ///
    /// # Safety
    /// Preconditions identical to [`Self::fft_in_place_with_executor`].
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
        let add_sub_kernel = ArchKernels::selected_add_sub_from_limbs_unchecked();
        let ctx = TransformContext::new(mod_bits, add_sub_kernel, executor);
        let cl = ctx.cl.get();
        let half_len = transform_len >> 1;
        // SAFETY: half_len*cl is one half of the complete validated matrix.
        let half_matrix_len = unsafe { half_len.unchecked_mul(cl) };
        // SAFETY: the validated matrix contains exactly two halves of
        // `half_len` complete coefficients.
        let (low_matrix, high_matrix) = unsafe { matrix.split_at_mut_unchecked(half_matrix_len) };
        // SAFETY: the parent's principal exponent times its length is period;
        // this child's doubled exponent is at most that representable period.
        let next_root = unsafe { root_shift.unchecked_mul(2) };
        // The child subtree sweeps each coefficient once per remaining
        // radix-2 level; the fork is priced by that enclosed work.
        #[expect(
            clippy::as_conversions,
            reason = "the logarithm of a positive usize power of two is below usize::BITS"
        )]
        let child_levels = half_len.trailing_zeros() as usize;
        // SAFETY: log2(half_len)<=half_len and all of its cl-limb slots exist.
        let child_work = unsafe { cl.unchecked_mul(child_levels) };
        // SAFETY: both halves are complete `half_len * cl` coefficient spans
        // and `scratch` still holds at least `cl` limbs.
        if Self::should_parallelize(
            half_len,
            child_work,
            cl,
            scratch.len(),
            executor.parallelism().get(),
        ) {
            // SAFETY: the helper partitions scratch into private slots; the two
            // matrix halves are disjoint.
            unsafe {
                Self::recurse_dif_pair(
                    low_matrix,
                    high_matrix,
                    half_len,
                    next_root,
                    scratch,
                    active_chunks,
                    active_chunks,
                    &ctx,
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
                    scratch,
                    active_chunks,
                    &ctx,
                );
                Self::fft_recursive_dif_with_executor(
                    high_matrix,
                    half_len,
                    next_root,
                    scratch,
                    active_chunks,
                    &ctx,
                );
            }
        }
    }

    /// Centralized grain and scratch policy for recursive transform splitting.
    ///
    /// The local child work sets the grain; `workers` only establishes whether
    /// parallel execution is available. Private scratch bounds the fork count.
    pub const fn should_parallelize(
        item_count: usize,
        unit_limb_work: usize,
        coeff_limbs: usize,
        scratch_len: usize,
        workers: usize,
    ) -> bool {
        Self::has_parallel_work(item_count, unit_limb_work, workers) && {
            // SAFETY: callers pass guarded widths of admitted SSA rings, whose
            // representable 4*bits and LIMB_BITS>=16 bound two coefficients.
            scratch_len >= unsafe { coeff_limbs.unchecked_mul(2) }
        }
    }

    /// Requires enough work in the smaller child of a binary fork.
    /// Descendants use their own range size rather than dividing local work by
    /// the full executor width while neighboring subtrees occupy the same pool.
    pub const fn has_parallel_work(
        item_count: usize,
        unit_limb_work: usize,
        workers: usize,
    ) -> bool {
        if workers <= 1 || item_count < 2 {
            return false;
        }
        item_count.div_euclid(2).saturating_mul(unit_limb_work) >= SSA_PARALLEL_MIN_LIMB_WORK
    }

    /// A radix-4 level can expose two independent child pairs when four private
    /// coefficient slots are available. This is a structural scratch condition,
    /// not a machine-specific grain threshold.
    pub const fn can_fork_four(coeff_limbs: usize, scratch_len: usize) -> bool {
        // SAFETY: callers pass guarded widths of admitted SSA rings, whose
        // representable 4*bits and LIMB_BITS>=16 bound four coefficients.
        let four_slots = unsafe { coeff_limbs.unchecked_mul(4) };
        scratch_len >= four_slots
    }

    /// Copies and twists one active coefficient range into its disjoint high half.
    ///
    /// # Safety
    /// `low_matrix` and `high_matrix` must each contain at least `count` complete
    /// `coeff_limbs(mod_bits)`-limb coefficients. Their active spans must not alias.
    /// The principal sparse-half range satisfies
    /// `start_twiddle + count*root_shift <= mod_bits`.
    #[inline]
    pub unsafe fn scatter_twiddle_range(
        low_matrix: &mut [Limb],
        high_matrix: &mut [Limb],
        count: usize,
        start_twiddle: usize,
        root_shift: usize,
        mod_bits: usize,
    ) {
        let cl = SsaRing::coeff_limbs(mod_bits).get();
        let mut twiddle_shift = start_twiddle;

        for i in 0..count {
            // SAFETY: the caller proves both matrices contain `count` complete
            // coefficients, so this exact cl-limb slot is in bounds.
            let (low_slot, high_slot) = unsafe {
                let offset = i.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                (
                    low_matrix.get_unchecked_mut(offset..end),
                    high_matrix.get_unchecked_mut(offset..end),
                )
            };
            if twiddle_shift == 0 {
                high_slot.copy_from_slice(low_slot);
            } else {
                // SAFETY: the caller proves the active matrix spans do not alias;
                // both slots contain exactly cl limbs and twiddle_shift is reduced.
                unsafe {
                    SsaRing::shift_from(high_slot, low_slot, twiddle_shift, mod_bits);
                }
            }
            // SAFETY: advancing to this range's inclusive endpoint reaches
            // at most mod_bits, below the whole-bit period 2*mod_bits.
            twiddle_shift = unsafe { twiddle_shift.unchecked_add(root_shift) };
        }
    }
}
