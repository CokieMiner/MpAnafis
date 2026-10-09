//! Cache-oblivious decimation-in-frequency recursion and radix-4 stages.

#![expect(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::{num::NonZeroUsize, ptr::from_mut};

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaRing, SsaTransform, TransformContext};

impl SsaTransform {
    /// Transforms a nonempty active prefix, treating the remaining slots as zero.
    ///
    /// # Safety
    /// The initialized matrix contains exactly `transform_len` complete slots
    /// for a positive limb-aligned ring; only its `active_len` prefix must be
    /// semi-normalized. The length is a power of two and its principal root
    /// satisfies `root_shift*transform_len=2*ctx.mod_bits`,
    /// and private scratch contains at least one coefficient. Sparse stages
    /// never read the physical tail. An empty support performs no writes;
    /// callers requesting an all-zero spectrum must establish it themselves.
    #[expect(
        clippy::too_many_lines,
        reason = "DIF FFT recursion manages bifurcated radix-4 stages, twiddle scheduling, and leaf dispatch"
    )]
    pub unsafe fn fft_recursive_dif_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        scratch: &mut [Limb],
        active_len: usize,
        ctx: &TransformContext<'_, E>,
    ) {
        if transform_len < 2 || active_len == 0 {
            return;
        }
        let cl = ctx.cl.get();
        let mod_bits = ctx.mod_bits;
        let add_sub_kernel = ctx.kernel;
        let executor = ctx.executor;

        if transform_len == 2 {
            // SAFETY: matrix has 2*cl limbs.
            let (low_slot, high_slot) = unsafe { matrix.split_at_mut_unchecked(cl) };
            if active_len == 1 {
                // High slot is zero initially: low + 0 = low, (low - 0)*1 = low.
                // SAFETY: high_slot has cl limbs.
                let dst_high = unsafe { high_slot.get_unchecked_mut(..cl) };
                // SAFETY: low_slot has cl limbs.
                let src_low = unsafe { low_slot.get_unchecked(..cl) };
                dst_high.copy_from_slice(src_low);
            } else {
                let high_dest = from_mut::<[Limb]>(high_slot);
                let high_source = high_dest.cast::<Limb>().cast_const();
                // SAFETY: low_slot and high_slot are disjoint cl-limb spans.
                unsafe {
                    SsaRing::add_sub(low_slot, high_dest, high_source, mod_bits, add_sub_kernel);
                }
            }
            return;
        }

        let quarter_len = transform_len >> 2;
        // SAFETY: the smaller codelets returned. The admitted power-of-two
        // transform_len is therefore >=4 and every quarter contains a slot.
        let quarter_width = unsafe { NonZeroUsize::new_unchecked(quarter_len) };
        // SAFETY: quarter_len*cl is one quarter of the complete matrix span.
        let quarter_matrix_len = unsafe { quarter_len.unchecked_mul(cl) };
        // SAFETY: `transform_len` is a power of two >= 4 and the recursive
        // contract gives four complete quarter matrices.
        let (q01, q23) =
            unsafe { matrix.split_at_mut_unchecked(quarter_matrix_len.unchecked_mul(2)) };
        // SAFETY: each parent pair contains exactly two complete quarter matrices.
        let (q0, q1) = unsafe { q01.split_at_mut_unchecked(quarter_matrix_len) };
        // SAFETY: q23 has the same validated width as q01.
        let (q2, q3) = unsafe { q23.split_at_mut_unchecked(quarter_matrix_len) };

        // SAFETY: q0, q1, q2, q3 are disjoint quarters and scratch has cl limbs.
        unsafe {
            Self::dif_radix4_stage(
                [q0, q1, q2, q3],
                quarter_width,
                root_shift,
                scratch,
                active_len,
                ctx,
            );
        }

        if transform_len == 4 {
            return;
        }

        if transform_len == 8 {
            // Eight-point DIF codelet: one radix-4 stage followed by four radix-2 butterflies.
            for q in [q0, q1, q2, q3] {
                // SAFETY: each q is a complete two-coefficient matrix at this
                // codelet boundary.
                let (low_slot, high_slot) = unsafe { q.split_at_mut_unchecked(cl) };
                if active_len == 1 {
                    // SAFETY: high_slot has cl limbs.
                    let dst_high = unsafe { high_slot.get_unchecked_mut(..cl) };
                    // SAFETY: low_slot has cl limbs.
                    let src_low = unsafe { low_slot.get_unchecked(..cl) };
                    dst_high.copy_from_slice(src_low);
                } else {
                    let high_dest = from_mut::<[Limb]>(high_slot);
                    let high_source = high_dest.cast::<Limb>().cast_const();
                    // SAFETY: low_slot and high_slot are disjoint cl-limb spans.
                    unsafe {
                        SsaRing::add_sub(
                            low_slot,
                            high_dest,
                            high_source,
                            mod_bits,
                            add_sub_kernel,
                        );
                    }
                }
            }
            return;
        }

        // SAFETY: transform_len>=16 here and root_shift*transform_len=period,
        // so 4*root_shift<period and is the child's principal exponent.
        let next_root = unsafe { root_shift.unchecked_mul(4) };
        let sub_active = active_len.min(quarter_len);
        // Each quarter subtree sweeps its coefficients once per remaining
        // radix-2 level; the fork is priced by that enclosed work.
        #[expect(
            clippy::as_conversions,
            reason = "the logarithm of a positive usize power of two is below usize::BITS and fits every pointer width"
        )]
        let child_levels = quarter_len.trailing_zeros() as usize;
        // SAFETY: log2(quarter_len)<=quarter_len, whose cl-limb slots form
        // a complete representable quarter of the supplied matrix.
        let child_work = unsafe { cl.unchecked_mul(child_levels) };

        if Self::should_parallelize(
            transform_len,
            child_work,
            cl,
            scratch.len(),
            executor.parallelism().get(),
        ) {
            if Self::can_fork_four(cl, scratch.len()) {
                let split = scratch.len().div_euclid(2);
                // SAFETY: `can_fork_four` and `should_parallelize` established an
                // arena containing two complete private scratch partitions.
                let (first_scratch, second_scratch) =
                    unsafe { scratch.split_at_mut_unchecked(split) };
                // SAFETY: the quarter pairs and their scratch arenas are disjoint.
                let ((), ()) = executor.join(
                    // SAFETY: q0/q1 and first_scratch are one disjoint recursion branch.
                    || unsafe {
                        Self::recurse_dif_pair(
                            q0,
                            q1,
                            quarter_len,
                            next_root,
                            first_scratch,
                            sub_active,
                            sub_active,
                            ctx,
                        );
                    },
                    // SAFETY: q2/q3 and second_scratch are the other disjoint branch.
                    || unsafe {
                        Self::recurse_dif_pair(
                            q2,
                            q3,
                            quarter_len,
                            next_root,
                            second_scratch,
                            sub_active,
                            sub_active,
                            ctx,
                        );
                    },
                );
            } else {
                // SAFETY: the helper partitions scratch into two private slots for
                // each fork and both quarter pairs are disjoint.
                unsafe {
                    Self::recurse_dif_pair(
                        q0,
                        q1,
                        quarter_len,
                        next_root,
                        scratch,
                        sub_active,
                        sub_active,
                        ctx,
                    );
                    Self::recurse_dif_pair(
                        q2,
                        q3,
                        quarter_len,
                        next_root,
                        scratch,
                        sub_active,
                        sub_active,
                        ctx,
                    );
                }
            }
        } else {
            // SAFETY: each quarter is a complete range and the sequential executor
            // path may reuse the one scratch slot after each child returns.
            unsafe {
                Self::fft_recursive_dif_with_executor(
                    q0,
                    quarter_len,
                    next_root,
                    scratch,
                    sub_active,
                    ctx,
                );
                Self::fft_recursive_dif_with_executor(
                    q1,
                    quarter_len,
                    next_root,
                    scratch,
                    sub_active,
                    ctx,
                );
                Self::fft_recursive_dif_with_executor(
                    q2,
                    quarter_len,
                    next_root,
                    scratch,
                    sub_active,
                    ctx,
                );
                Self::fft_recursive_dif_with_executor(
                    q3,
                    quarter_len,
                    next_root,
                    scratch,
                    sub_active,
                    ctx,
                );
            }
        }
    }

    /// Forks two independent DIF child ranges while giving each branch one private
    /// twiddle slot. The same helper is reused for both radix-4 child pairs.
    #[expect(
        clippy::too_many_arguments,
        reason = "bifurcated DIF recursion passes split matrix pairs, active bounds, and transform context"
    )]
    pub unsafe fn recurse_dif_pair<E: ParallelExecutor>(
        first: &mut [Limb],
        second: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        scratch: &mut [Limb],
        active_len_first: usize,
        active_len_second: usize,
        ctx: &TransformContext<'_, E>,
    ) {
        // `should_parallelize` proved the arena holds at least two coefficient
        // slots before this helper was entered, so both halves hold one slot.
        let split = scratch.len().div_euclid(2);
        // SAFETY: `should_parallelize` established two complete cl-limb scratch
        // slots, so the computed half split is within the validated arena.
        let (scratch_left, scratch_right) = unsafe { scratch.split_at_mut_unchecked(split) };
        let ((), ()) = ctx.executor.join(
            || {
                // SAFETY: first and scratch_left are disjoint complete ranges.
                unsafe {
                    Self::fft_recursive_dif_with_executor(
                        first,
                        transform_len,
                        root_shift,
                        scratch_left,
                        active_len_first,
                        ctx,
                    );
                }
            },
            || {
                // SAFETY: second and scratch_right are disjoint complete ranges.
                unsafe {
                    Self::fft_recursive_dif_with_executor(
                        second,
                        transform_len,
                        root_shift,
                        scratch_right,
                        active_len_second,
                        ctx,
                    );
                }
            },
        );
    }
}
