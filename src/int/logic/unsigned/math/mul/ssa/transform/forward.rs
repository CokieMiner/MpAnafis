//! Cache-oblivious decimation-in-frequency recursion and radix-4 stages.

#![allow(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::ptr::from_mut;

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaRing, SsaTransform};

impl SsaTransform {
    /// Forks two independent DIF child ranges while giving each branch one private
    /// twiddle slot. The same helper is reused for both radix-4 child pairs.
    #[allow(
        clippy::too_many_arguments,
        reason = "A recursive child pair needs its active widths, range, root, arithmetic kernel, executor, and scratch"
    )]
    pub(crate) unsafe fn recurse_dif_pair<E: ParallelExecutor>(
        first: &mut [Limb],
        second: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
        add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
        active_len_first: usize,
        active_len_second: usize,
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
                    Self::fft_recursive_dif_with_executor(
                        first,
                        transform_len,
                        root_shift,
                        mod_bits,
                        scratch_left,
                        add_sub_kernel,
                        active_len_first,
                        executor,
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
                        mod_bits,
                        scratch_right,
                        add_sub_kernel,
                        active_len_second,
                        executor,
                    );
                }
            },
        );
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "The recursive DIF worker keeps radix-4 staging, the eight-point codelet, and fork/join partitioning in one cache-local kernel"
    )]
    pub(crate) unsafe fn fft_recursive_dif_with_executor<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
        add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
        active_len: usize,
        executor: &E,
    ) {
        if transform_len < 2 || active_len == 0 {
            return;
        }
        let cl = SsaRing::coeff_limbs(mod_bits);
        let period = mod_bits.wrapping_mul(2);

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
        let quarter_matrix_len = quarter_len.wrapping_mul(cl);
        // SAFETY: `transform_len` is a power of two >= 4 and the recursive
        // contract gives four complete quarter matrices.
        let (q01, q23) =
            unsafe { matrix.split_at_mut_unchecked(quarter_matrix_len.wrapping_mul(2)) };
        // SAFETY: each parent pair contains exactly two complete quarter matrices.
        let (q0, q1) = unsafe { q01.split_at_mut_unchecked(quarter_matrix_len) };
        // SAFETY: q23 has the same validated width as q01.
        let (q2, q3) = unsafe { q23.split_at_mut_unchecked(quarter_matrix_len) };

        // SAFETY: q0, q1, q2, q3 are disjoint quarters and scratch has cl limbs.
        unsafe {
            dif_radix4_stage(
                [q0, q1, q2, q3],
                root_shift,
                mod_bits,
                scratch,
                add_sub_kernel,
                active_len,
            );
        }

        if transform_len == 4 {
            return;
        }

        if transform_len == 8 {
            // Direct 8-point radix-8 DIF codelet: 4-point radix-4 stage followed by 4 radix-2 butterflies.
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

        let next_root = SsaRing::reduce_mod_period(root_shift.wrapping_mul(4), period);
        let sub_active = active_len.min(quarter_len);

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
                        Self::recurse_dif_pair(
                            q0,
                            q1,
                            quarter_len,
                            next_root,
                            mod_bits,
                            first_scratch,
                            add_sub_kernel,
                            sub_active,
                            sub_active,
                            executor,
                        );
                    },
                    // SAFETY: q2/q3 and second_scratch are the other disjoint branch.
                    || unsafe {
                        Self::recurse_dif_pair(
                            q2,
                            q3,
                            quarter_len,
                            next_root,
                            mod_bits,
                            second_scratch,
                            add_sub_kernel,
                            sub_active,
                            sub_active,
                            executor,
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
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        sub_active,
                        sub_active,
                        executor,
                    );
                    Self::recurse_dif_pair(
                        q2,
                        q3,
                        quarter_len,
                        next_root,
                        mod_bits,
                        scratch,
                        add_sub_kernel,
                        sub_active,
                        sub_active,
                        executor,
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
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_active,
                    executor,
                );
                Self::fft_recursive_dif_with_executor(
                    q1,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_active,
                    executor,
                );
                Self::fft_recursive_dif_with_executor(
                    q2,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_active,
                    executor,
                );
                Self::fft_recursive_dif_with_executor(
                    q3,
                    quarter_len,
                    next_root,
                    mod_bits,
                    scratch,
                    add_sub_kernel,
                    sub_active,
                    executor,
                );
            }
        }
    }
}

/// Computes a single radix-4 DIF pass across 4 disjoint quarters.
///
/// # Safety
/// All 4 quarter slices have length `quarter_len * coeff_limbs(mod_bits)` and are disjoint.
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "Radix-4 DIF stage fuses two unrolled butterfly stages with strided coefficient prefetching"
)]
unsafe fn dif_radix4_stage(
    quarters: [&mut [Limb]; 4],
    root_shift: usize,
    mod_bits: usize,
    scratch: &mut [Limb],
    add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
    active_len: usize,
) {
    let [q0, q1, q2, q3] = quarters;
    let cl = SsaRing::coeff_limbs(mod_bits);
    let quarter_len = q0.len().div_euclid(cl);
    let period = mod_bits.wrapping_mul(2);

    if active_len <= quarter_len {
        let mut twiddle_shift = 0_usize;
        for i in 0..active_len {
            let offset = i.wrapping_mul(cl);
            // SAFETY: i < active_len <= quarter_len ensures in-bounds for all 4 disjoint quarters.
            let (u0, u1, u2, u3) = unsafe {
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

            if w1 == 0 {
                u2.copy_from_slice(u0);
            } else {
                // SAFETY: u2 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u2, u0, w1, mod_bits);
                }
            }

            if w2 == 0 {
                u1.copy_from_slice(u0);
            } else {
                // SAFETY: u1 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u1, u0, w2, mod_bits);
                }
            }

            if w3 == 0 {
                u3.copy_from_slice(u0);
            } else {
                // SAFETY: u3 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u3, u0, w3, mod_bits);
                }
            }

            twiddle_shift = twiddle_shift.wrapping_add(root_shift);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
        }
        return;
    }

    let q1_chunks = q1.chunks_exact_mut(cl);
    let q2_chunks = q2.chunks_exact_mut(cl);
    let q3_chunks = q3.chunks_exact_mut(cl);
    let i_shift = SsaRing::reduce_mod_period(root_shift.wrapping_mul(quarter_len), period);
    let mut twiddle_shift = 0_usize;
    // SAFETY: the caller contract requires at least one initialized cl-limb
    // scratch slot; the range is therefore in bounds and never empty.
    let scratch_slot = unsafe { scratch.get_unchecked_mut(..cl) };

    if active_len >= quarter_len.wrapping_mul(4) {
        for (((u0, u1), u2), u3) in q0
            .chunks_exact_mut(cl)
            .zip(q1_chunks)
            .zip(q2_chunks)
            .zip(q3_chunks)
        {
            // Stage 1: Butterfly pairs (u0, u2) and (u1, u3)
            let u2_dest = from_mut::<[Limb]>(u2);
            let u2_src = u2_dest.cast::<Limb>().cast_const();
            let u3_dest = from_mut::<[Limb]>(u3);
            let u3_src = u3_dest.cast::<Limb>().cast_const();
            // SAFETY: u0, u1, u2, u3 are disjoint cl-limb slices in distinct quarters.
            unsafe {
                SsaRing::add_sub(u0, u2_dest, u2_src, mod_bits, add_sub_kernel);
                SsaRing::add_sub(u1, u3_dest, u3_src, mod_bits, add_sub_kernel);
            }

            // Stage 2: Combine (u0, u1) -> v0 in u0, v1 in u1
            let u1_dest = from_mut::<[Limb]>(u1);
            let u1_src = u1_dest.cast::<Limb>().cast_const();
            // SAFETY: u0 and u1 are disjoint cl-limb slices.
            unsafe {
                SsaRing::add_sub(u0, u1_dest, u1_src, mod_bits, add_sub_kernel);
            }
            let w2 = SsaRing::reduce_mod_period(twiddle_shift.wrapping_mul(2), period);
            if w2 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(u1, w2, mod_bits, scratch_slot);
                }
            }

            // Stage 3: Combine (u2, u3) with i_shift -> v2 in u2, v3 in u3
            let u3_operand_ptr = if i_shift == 0 {
                u3.as_ptr()
            } else {
                // SAFETY: scratch_slot and u3 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(scratch_slot, u3, i_shift, mod_bits);
                }
                scratch_slot.as_ptr()
            };

            let u3_diff_dest = from_mut::<[Limb]>(u3);
            // SAFETY: u2, u3, u3_operand_ptr are disjoint cl-limb spans.
            unsafe {
                SsaRing::add_sub(u2, u3_diff_dest, u3_operand_ptr, mod_bits, add_sub_kernel);
            }

            let w1 = twiddle_shift;
            if w1 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(u2, w1, mod_bits, scratch_slot);
                }
            }

            let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);
            if w3 != 0 {
                // SAFETY: scratch_slot is a disjoint cl-limb staging span.
                unsafe {
                    SsaRing::shift_in_place(u3, w3, mod_bits, scratch_slot);
                }
            }

            twiddle_shift = twiddle_shift.wrapping_add(root_shift);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
        }
        return;
    }

    let q = quarter_len;
    for (i, (((u0, u1), u2), u3)) in q0
        .chunks_exact_mut(cl)
        .zip(q1_chunks)
        .zip(q2_chunks)
        .zip(q3_chunks)
        .enumerate()
    {
        let is_q3_active = i.wrapping_add(q.wrapping_mul(3)) < active_len;
        let is_q2_active = i.wrapping_add(q.wrapping_mul(2)) < active_len;
        let is_q1_active = i.wrapping_add(q) < active_len;
        let is_q0_active = i < active_len;

        if !is_q0_active {
            twiddle_shift = twiddle_shift.wrapping_add(root_shift);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
            continue;
        }

        if !is_q1_active {
            // u1 = u2 = u3 = 0 -> direct twiddle copy from u0
            let w1 = twiddle_shift;
            let w2 = SsaRing::reduce_mod_period(w1.wrapping_mul(2), period);
            let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);

            if w1 == 0 {
                u2.copy_from_slice(u0);
            } else {
                // SAFETY: u2 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u2, u0, w1, mod_bits);
                }
            }
            if w2 == 0 {
                u1.copy_from_slice(u0);
            } else {
                // SAFETY: u1 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u1, u0, w2, mod_bits);
                }
            }
            if w3 == 0 {
                u3.copy_from_slice(u0);
            } else {
                // SAFETY: u3 and u0 are disjoint cl-limb spans.
                unsafe {
                    SsaRing::shift_from(u3, u0, w3, mod_bits);
                }
            }
            twiddle_shift = twiddle_shift.wrapping_add(root_shift);
            if twiddle_shift >= period {
                twiddle_shift = twiddle_shift.wrapping_sub(period);
            }
            continue;
        }

        // Stage 1: Butterfly pairs (u0, u2) and (u1, u3)
        if is_q2_active {
            let u2_dest = from_mut::<[Limb]>(u2);
            let u2_src = u2_dest.cast::<Limb>().cast_const();
            // SAFETY: u0 and u2 are disjoint cl-limb slices in distinct quarters.
            unsafe {
                SsaRing::add_sub(u0, u2_dest, u2_src, mod_bits, add_sub_kernel);
            }
        } else {
            // u2 is zero: u0 + 0 = u0, u0 - 0 = u0 -> copy u0 into u2
            u2.copy_from_slice(u0);
        }

        if is_q3_active {
            let u3_dest = from_mut::<[Limb]>(u3);
            let u3_src = u3_dest.cast::<Limb>().cast_const();
            // SAFETY: u1 and u3 are disjoint cl-limb slices in distinct quarters.
            unsafe {
                SsaRing::add_sub(u1, u3_dest, u3_src, mod_bits, add_sub_kernel);
            }
        } else {
            // u3 is zero: u1 + 0 = u1, u1 - 0 = u1 -> copy u1 into u3
            u3.copy_from_slice(u1);
        }

        // Stage 2: Combine (u0, u1) -> v0 in u0, v1 in u1
        let u1_dest = from_mut::<[Limb]>(u1);
        let u1_src = u1_dest.cast::<Limb>().cast_const();
        // SAFETY: u0 and u1 are disjoint cl-limb slices.
        unsafe {
            SsaRing::add_sub(u0, u1_dest, u1_src, mod_bits, add_sub_kernel);
        }
        let w2 = SsaRing::reduce_mod_period(twiddle_shift.wrapping_mul(2), period);
        if w2 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(u1, w2, mod_bits, scratch_slot);
            }
        }

        // Stage 3: Combine (u2, u3) with i_shift -> v2 in u2, v3 in u3
        let u3_operand_ptr = if i_shift == 0 {
            u3.as_ptr()
        } else {
            // SAFETY: scratch_slot and u3 are disjoint cl-limb spans.
            unsafe {
                SsaRing::shift_from(scratch_slot, u3, i_shift, mod_bits);
            }
            scratch_slot.as_ptr()
        };

        let u3_diff_dest = from_mut::<[Limb]>(u3);
        // SAFETY: u2, u3, u3_operand_ptr are disjoint cl-limb spans.
        unsafe {
            SsaRing::add_sub(u2, u3_diff_dest, u3_operand_ptr, mod_bits, add_sub_kernel);
        }

        let w1 = twiddle_shift;
        if w1 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(u2, w1, mod_bits, scratch_slot);
            }
        }

        let w3 = SsaRing::reduce_mod_period(w1.wrapping_add(w2), period);
        if w3 != 0 {
            // SAFETY: scratch_slot is a disjoint cl-limb staging span.
            unsafe {
                SsaRing::shift_in_place(u3, w3, mod_bits, scratch_slot);
            }
        }

        twiddle_shift = twiddle_shift.wrapping_add(root_shift);
        if twiddle_shift >= period {
            twiddle_shift = twiddle_shift.wrapping_sub(period);
        }
    }
}
