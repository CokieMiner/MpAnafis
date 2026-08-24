//! Cache-oblivious decimation-in-time recursion and radix-4 stages.

#![allow(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::ptr::from_mut;

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaRing, SsaTransform};

impl SsaTransform {
    /// Forks two independent DIT child ranges while giving each branch one private
    /// twiddle slot. The same helper is reused for both radix-4 child pairs.
    #[allow(
        clippy::too_many_arguments,
        reason = "A recursive child pair needs its active output widths, range, root, arithmetic kernel, executor, and scratch"
    )]
    pub(crate) unsafe fn recurse_dit_pair<E: ParallelExecutor>(
        first: &mut [Limb],
        second: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
        add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
        needed_out_first: usize,
        needed_out_second: usize,
        executor: &E,
    ) {
        // `should_parallelize` proved the arena holds at least two coefficient
        // slots before this helper was entered, so both halves hold one slot.
        let split = scratch.len().div_euclid(2);
        // SAFETY: `should_parallelize` established two complete cl-limb scratch
        // slots, so the computed half split is within the validated arena.
        let (scratch_left, scratch_right) = unsafe { scratch.split_at_mut_unchecked(split) };
        let ((), ()) = executor.join(
            || {
                // SAFETY: first and scratch_left are disjoint complete ranges.
                unsafe {
                    Self::fft_recursive_dit_with_executor(
                        first,
                        transform_len,
                        root_shift,
                        mod_bits,
                        scratch_left,
                        add_sub_kernel,
                        needed_out_first,
                        executor,
                    );
                }
            },
            || {
                // SAFETY: second and scratch_right are disjoint complete ranges.
                unsafe {
                    Self::fft_recursive_dit_with_executor(
                        second,
                        transform_len,
                        root_shift,
                        mod_bits,
                        scratch_right,
                        add_sub_kernel,
                        needed_out_second,
                        executor,
                    );
                }
            },
        );
    }

    /// Mirrors [`fft_recursive_dif_with_executor`]: the 4 contiguous quarters are transformed
    /// first and combined afterwards in radix-4 butterflies.
    ///
    /// # Safety
    /// Identical to [`fft_recursive_dif_with_executor`].
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "The inverse radix-4 recursion keeps child scheduling and the final stage together"
    )]
    pub(crate) unsafe fn fft_recursive_dit_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
        add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
        needed_out: usize,
        executor: &E,
    ) {
        if transform_len < 2 || needed_out == 0 {
            return;
        }
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);

        if transform_len == 2 {
            // SAFETY: the recursive contract gives exactly two initialized cl-limb
            // coefficient slots for this base case.
            let (low_slot, high_slot) = unsafe { matrix.split_at_mut_unchecked(cl) };
            let high_dest = from_mut::<[Limb]>(high_slot);
            let high_source = high_dest.cast::<Limb>().cast_const();
            // SAFETY: low_slot and high_slot are disjoint cl-limb spans.
            unsafe {
                SsaRing::add_sub(low_slot, high_dest, high_source, mod_bits, add_sub_kernel);
            }
            return;
        }

        let quarter_len = transform_len >> 2;
        let quarter_matrix_len = quarter_len.wrapping_mul(cl);
        // SAFETY: `transform_len` is a power of two >= 4 and the recursive
        // contract gives four complete quarter matrices.
        let (q01, q23) =
            unsafe { matrix.split_at_mut_unchecked(quarter_matrix_len.wrapping_mul(2)) };
        // SAFETY: each parent pair contains exactly two complete quarter matrices.
        let (q0, q1) = unsafe { q01.split_at_mut_unchecked(quarter_matrix_len) };
        // SAFETY: q23 has the same validated width as q01.
        let (q2, q3) = unsafe { q23.split_at_mut_unchecked(quarter_matrix_len) };

        if transform_len == 4 {
            // SAFETY: q0, q1, q2, q3 are disjoint quarters and scratch has cl limbs.
            unsafe {
                dit_radix4_stage(
                    [q0, q1, q2, q3],
                    root_shift,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    needed_out,
                );
            }
            return;
        }

        if transform_len == 8 {
            // Direct 8-point radix-8 DIT codelet: 4 radix-2 butterflies followed by 4-point radix-4 stage.
            for q in [&mut *q0, &mut *q1, &mut *q2, &mut *q3] {
                // SAFETY: each q is a complete two-coefficient matrix at this
                // codelet boundary.
                let (low_slot, high_slot) = unsafe { q.split_at_mut_unchecked(cl) };
                let high_dest = from_mut::<[Limb]>(high_slot);
                let high_source = high_dest.cast::<Limb>().cast_const();
                // SAFETY: low_slot and high_slot are disjoint cl-limb spans.
                unsafe {
                    SsaRing::add_sub(low_slot, high_dest, high_source, mod_bits, add_sub_kernel);
                }
            }
            // SAFETY: q0, q1, q2, q3 are disjoint quarters and scratch has cl limbs.
            unsafe {
                dit_radix4_stage(
                    [q0, q1, q2, q3],
                    root_shift,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    needed_out,
                );
            }
            return;
        }

        let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(4), period);
        let sub_needed = needed_out.min(quarter_len);

        if Self::should_parallelize(quarter_len, cl, scratch.len(), executor) {
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
                        Self::recurse_dit_pair(
                            q0,
                            q1,
                            quarter_len,
                            next_root,
                            mod_bits,
                            first_scratch,
                            add_sub_kernel,
                            sub_needed,
                            sub_needed,
                            executor,
                        );
                    },
                    // SAFETY: q2/q3 and second_scratch are the other disjoint branch.
                    || unsafe {
                        Self::recurse_dit_pair(
                            q2,
                            q3,
                            quarter_len,
                            next_root,
                            mod_bits,
                            second_scratch,
                            add_sub_kernel,
                            sub_needed,
                            sub_needed,
                            executor,
                        );
                    },
                );
            } else {
                // SAFETY: the helper partitions scratch into two private slots for
                // each fork and both quarter pairs are disjoint.
                unsafe {
                    Self::recurse_dit_pair(
                        q0,
                        q1,
                        quarter_len,
                        next_root,
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        sub_needed,
                        sub_needed,
                        executor,
                    );
                    Self::recurse_dit_pair(
                        q2,
                        q3,
                        quarter_len,
                        next_root,
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        sub_needed,
                        sub_needed,
                        executor,
                    );
                }
            }
        } else {
            // SAFETY: each quarter is a complete range and the sequential executor
            // path may reuse the one scratch slot after each child returns.
            unsafe {
                Self::fft_recursive_dit_with_executor(
                    q0,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_needed,
                    executor,
                );
                Self::fft_recursive_dit_with_executor(
                    q1,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_needed,
                    executor,
                );
                Self::fft_recursive_dit_with_executor(
                    q2,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_needed,
                    executor,
                );
                Self::fft_recursive_dit_with_executor(
                    q3,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_needed,
                    executor,
                );
            }
        }

        // SAFETY: q0, q1, q2, q3 are disjoint quarters and scratch has cl limbs.
        unsafe {
            dit_radix4_stage(
                [q0, q1, q2, q3],
                root_shift,
                mod_bits,
                scratch,
                add_sub_kernel,
                needed_out,
            );
        }
    }
}

/// Computes a single radix-4 DIT pass across 4 disjoint quarters.
///
/// # Safety
/// All 4 quarter slices have length `quarter_len * coeff_limbs(mod_bits)` and are disjoint.
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "Radix-4 DIT stage fuses two unrolled butterfly stages with strided coefficient prefetching"
)]
unsafe fn dit_radix4_stage(
    quarters: [&mut [Limb]; 4],
    root_shift: usize,
    mod_bits: usize,
    scratch: &mut [Limb],
    add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
    needed_out: usize,
) {
    let [q0, q1, q2, q3] = quarters;
    let cl = SsaRing::coeff_limbs(mod_bits);
    let quarter_len = q0.len().div_euclid(cl);
    let period = mod_bits.wrapping_mul(2);
    let twiddle_step = period.wrapping_sub(root_shift);
    let mut twiddle_shift = 0_usize;
    // SAFETY: the caller contract requires at least one initialized cl-limb
    // scratch slot; the range is therefore in bounds and never empty.
    let scratch_slot = unsafe { scratch.get_unchecked_mut(..cl) };

    if needed_out <= quarter_len {
        for i in 0..needed_out {
            let offset = i.wrapping_mul(cl);
            // SAFETY: i < needed_out <= quarter_len ensures in-bounds for all 4 disjoint quarters.
            let (v0, v1, v2, v3) = unsafe {
                (
                    q0.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    q1.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    q2.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                    q3.get_unchecked_mut(offset..offset.wrapping_add(cl)),
                )
            };

            let w1 = twiddle_shift;
            let w2 = SsaRing::reduce_mod_period(w1.wrapping_mul(2), period);
            let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);

            if w2 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v1, w2, mod_bits, scratch_slot);
                }
            }

            if w1 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v2, w1, mod_bits, scratch_slot);
                }
            }

            if w3 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v3, w3, mod_bits, scratch_slot);
                }
            }

            // Combine (v0, v1) -> a0 in v0
            let v1_dest = from_mut::<[Limb]>(v1);
            let v1_src = v1_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v1 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v1_dest, v1_src, mod_bits, add_sub_kernel);
            }

            // Combine (v2, v3) -> b0 in v2
            let v3_dest = from_mut::<[Limb]>(v3);
            let v3_src = v3_dest.cast::<Limb>().cast_const();
            // SAFETY: v2 and v3 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v2, v3_dest, v3_src, mod_bits, add_sub_kernel);
            }

            // Final combination: u0 = a0 + b0 in v0
            let v2_dest = from_mut::<[Limb]>(v2);
            let v2_src = v2_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v2 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v2_dest, v2_src, mod_bits, add_sub_kernel);
            }

            twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
        }
        return;
    }

    let q0_chunks = q0.chunks_exact_mut(cl);
    let q1_chunks = q1.chunks_exact_mut(cl);
    let q2_chunks = q2.chunks_exact_mut(cl);
    let q3_chunks = q3.chunks_exact_mut(cl);
    let i_shift = SsaRing::reduce_mod_period(root_shift.wrapping_mul(quarter_len), period);
    let inv_i_shift = period.wrapping_sub(i_shift);

    if needed_out >= quarter_len.wrapping_mul(4) {
        for (((v0, v1), v2), v3) in q0_chunks.zip(q1_chunks).zip(q2_chunks).zip(q3_chunks) {
            // Untwiddle v1, v2, v3
            let w1 = twiddle_shift;
            let w2 = SsaRing::reduce_mod_period(w1.wrapping_mul(2), period);
            let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);

            if w2 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v1, w2, mod_bits, scratch_slot);
                }
            }

            if w1 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v2, w1, mod_bits, scratch_slot);
                }
            }

            if w3 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(v3, w3, mod_bits, scratch_slot);
                }
            }

            // Combine (v0, v1) -> a0 in v0, a1 in v1
            let v1_dest = from_mut::<[Limb]>(v1);
            let v1_src = v1_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v1 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v1_dest, v1_src, mod_bits, add_sub_kernel);
            }

            // Combine (v2, v3) -> b0 in v2, b1 in v3
            let v3_dest = from_mut::<[Limb]>(v3);
            let v3_src = v3_dest.cast::<Limb>().cast_const();
            // SAFETY: v2 and v3 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v2, v3_dest, v3_src, mod_bits, add_sub_kernel);
            }

            // Final combinations:
            // (a0, b0) -> u0 in v0, u2 in v2
            let v2_dest = from_mut::<[Limb]>(v2);
            let v2_src = v2_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v2 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v2_dest, v2_src, mod_bits, add_sub_kernel);
            }

            // Multiply b1 (in v3) by i^-1
            let v3_operand_ptr = if inv_i_shift == 0 {
                v3.as_ptr()
            } else {
                // SAFETY: scratch_slot and v3 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(scratch_slot, v3, inv_i_shift, mod_bits);
                }
                scratch_slot.as_ptr()
            };

            // (a1, i^-1 * b1) -> u1 in v1, u3 in v3
            let v3_diff_dest = from_mut::<[Limb]>(v3);
            // SAFETY: v1, v3, v3_operand_ptr are disjoint cl-limb spans.
            unsafe {
                SsaRing::add_sub(v1, v3_diff_dest, v3_operand_ptr, mod_bits, add_sub_kernel);
            }

            twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
        }
        return;
    }

    let q = quarter_len;
    for (i, (((v0, v1), v2), v3)) in q0_chunks
        .zip(q1_chunks)
        .zip(q2_chunks)
        .zip(q3_chunks)
        .enumerate()
    {
        let is_q1_needed = i.wrapping_add(q) < needed_out;
        let is_q0_needed = i < needed_out;

        if !is_q0_needed {
            twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
            continue;
        }

        // Untwiddle v1, v2, v3
        let w1 = twiddle_shift;
        let w2 = SsaRing::reduce_mod_period(w1.wrapping_mul(2), period);
        let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);

        if w2 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(v1, w2, mod_bits, scratch_slot);
            }
        }

        if w1 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(v2, w1, mod_bits, scratch_slot);
            }
        }

        if w3 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(v3, w3, mod_bits, scratch_slot);
            }
        }

        if !is_q1_needed {
            // Only v0 (q0) is needed: u0 = (v0 + v1') + (v2' + v3')
            let v1_dest = from_mut::<[Limb]>(v1);
            let v1_src = v1_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v1 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v1_dest, v1_src, mod_bits, add_sub_kernel);
            }
            let v3_dest = from_mut::<[Limb]>(v3);
            let v3_src = v3_dest.cast::<Limb>().cast_const();
            // SAFETY: v2 and v3 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v2, v3_dest, v3_src, mod_bits, add_sub_kernel);
            }
            let v2_dest = from_mut::<[Limb]>(v2);
            let v2_src = v2_dest.cast::<Limb>().cast_const();
            // SAFETY: v0 and v2 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(v0, v2_dest, v2_src, mod_bits, add_sub_kernel);
            }
            twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
            continue;
        }

        // Combine (v0, v1) -> a0 in v0, a1 in v1
        let v1_dest = from_mut::<[Limb]>(v1);
        let v1_src = v1_dest.cast::<Limb>().cast_const();
        // SAFETY: v0 and v1 are disjoint cl-limb slices.
        unsafe {
            SsaRing::add_sub(v0, v1_dest, v1_src, mod_bits, add_sub_kernel);
        }

        // Combine (v2, v3) -> b0 in v2, b1 in v3
        let v3_dest = from_mut::<[Limb]>(v3);
        let v3_src = v3_dest.cast::<Limb>().cast_const();
        // SAFETY: v2 and v3 are disjoint cl-limb slices.
        unsafe {
            SsaRing::add_sub(v2, v3_dest, v3_src, mod_bits, add_sub_kernel);
        }

        // Final combinations:
        // (a0, b0) -> u0 in v0, u2 in v2
        let v2_dest = from_mut::<[Limb]>(v2);
        let v2_src = v2_dest.cast::<Limb>().cast_const();
        // SAFETY: v0 and v2 are disjoint cl-limb slices.
        unsafe {
            SsaRing::add_sub(v0, v2_dest, v2_src, mod_bits, add_sub_kernel);
        }

        // Multiply b1 (in v3) by i^-1
        let v3_operand_ptr = if inv_i_shift == 0 {
            v3.as_ptr()
        } else {
            // SAFETY: scratch_slot and v3 are disjoint cl-limb spans.
            unsafe {
                SsaRing::shift_from(scratch_slot, v3, inv_i_shift, mod_bits);
            }
            scratch_slot.as_ptr()
        };

        // (a1, i^-1 * b1) -> u1 in v1, u3 in v3
        let v3_diff_dest = from_mut::<[Limb]>(v3);
        // SAFETY: v1, v3, v3_operand_ptr are disjoint cl-limb spans.
        unsafe {
            SsaRing::add_sub(v1, v3_diff_dest, v3_operand_ptr, mod_bits, add_sub_kernel);
        }

        twiddle_shift = twiddle_shift.wrapping_add(twiddle_step);
        if twiddle_shift >= period {
            twiddle_shift = twiddle_shift.wrapping_sub(period);
        }
    }
}
