//! Radix-4 decimation-in-frequency stages and streaming coefficient leaves.

#![expect(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::{
    num::NonZeroUsize,
    ptr::{copy_nonoverlapping, from_mut},
};

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaRing, SsaTransform, TransformContext};

impl SsaTransform {
    /// Computes a single radix-4 DIF pass across 4 disjoint quarters.
    ///
    /// With the principal root omega=2^(2n/K), quarter index j<K/4 gives
    /// j*root<n/2 and 3*j*root<3n/2. All three twiddles are already reduced;
    /// the radix-four imaginary unit is i=2^(n/2).
    ///
    /// # Safety
    /// All four disjoint quarter slices contain `quarter_width` complete slots
    /// for `ctx.mod_bits`. `root_shift*(4*quarter_width)=2*ctx.mod_bits`;
    /// scratch contains a complete coefficient and `active_len<=4*quarter_width`.
    #[expect(
        clippy::too_many_lines,
        reason = "Radix-4 DIF stage fuses sparse, dense, and reduced-twiddle paths with strided coefficient prefetching"
    )]
    pub unsafe fn dif_radix4_stage<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        quarter_width: NonZeroUsize,
        root_shift: usize,
        scratch: &mut [Limb],
        active_len: usize,
        ctx: &TransformContext<'_, E>,
    ) {
        let [q0, q1, q2, q3] = quarters;
        let cl = ctx.cl.get();
        let quarter_len = quarter_width.get();
        let mod_bits = ctx.mod_bits;
        let add_sub_kernel = ctx.kernel;

        if active_len <= quarter_len {
            if active_len == 0 {
                return;
            }
            // The zero-index twiddle is the identity. Peeling it leaves only
            // positive exponents in the streaming prefix below.
            // SAFETY: 0<active_len<=quarter_len gives a complete first slot in
            // every disjoint quarter, with the same cl-limb width.
            unsafe {
                let first = q0.get_unchecked(..cl);
                q2.get_unchecked_mut(..cl).copy_from_slice(first);
                q1.get_unchecked_mut(..cl).copy_from_slice(first);
                q3.get_unchecked_mut(..cl).copy_from_slice(first);
            }
            let mut twiddle_shift = root_shift;
            for i in 1..active_len {
                // SAFETY: i < active_len <= quarter_len ensures in-bounds for all 4 disjoint quarters.
                let (u0, u1, u2, u3) = unsafe {
                    let offset = i.unchecked_mul(cl);
                    let end = offset.unchecked_add(cl);
                    (
                        q0.get_unchecked_mut(offset..end),
                        q1.get_unchecked_mut(offset..end),
                        q2.get_unchecked_mut(offset..end),
                        q3.get_unchecked_mut(offset..end),
                    )
                };

                let w1 = twiddle_shift;
                // SAFETY: i<quarter_len gives w1<period/4, so both products
                // are below period and fit the validated ring exponent width.
                let (w2, w3) = unsafe { (w1.unchecked_mul(2), w1.unchecked_mul(3)) };

                // SAFETY: i>=1 and the principal root is positive. Thus w1,
                // w2, and w3 are positive reduced exponents. The cl-limb
                // source and three complete destination slots are disjoint.
                unsafe {
                    SsaRing::shift_from(u2, u0, w1, mod_bits);
                    SsaRing::shift_from(u1, u0, w2, mod_bits);
                    SsaRing::shift_from(u3, u0, w3, mod_bits);
                }

                // SAFETY: the inclusive next index is at most quarter_len,
                // so its principal twiddle is at most period/4.
                twiddle_shift = unsafe { twiddle_shift.unchecked_add(root_shift) };
            }
            return;
        }

        let i_shift = mod_bits >> 1;

        // SAFETY: four quarters form the complete validated transform length.
        let complete_len = unsafe { quarter_len.unchecked_mul(4) };
        if active_len >= complete_len {
            // SAFETY: quarters are disjoint, scratch has at least cl limbs.
            unsafe {
                Self::dif_radix4_dense_parallel(
                    [q0, q1, q2, q3],
                    quarter_width,
                    0,
                    root_shift,
                    i_shift,
                    scratch,
                    ctx,
                );
            }
            return;
        }

        // SAFETY: the caller contract requires at least one initialized cl-limb
        // scratch slot; the range is therefore in bounds and never empty.
        let scratch_slot = unsafe { scratch.get_unchecked_mut(..cl) };
        let q = quarter_len;
        // SAFETY: both offsets lie below the admitted complete 4*q matrix.
        let (q2_start, q3_start) = unsafe { (q.unchecked_mul(2), q.unchecked_mul(3)) };

        // Partial-support butterflies read q2 and q3 only when their input
        // support includes this slot. Their common arithmetic is independent
        // of the three output twiddles, allowing the identity to be peeled.
        let butterfly = move |u0: &mut [Limb],
                              u1: &mut [Limb],
                              u2: &mut [Limb],
                              u3: &mut [Limb],
                              staging: &mut [Limb],
                              has_q2: bool,
                              has_q3: bool| {
            if has_q2 {
                let difference = from_mut::<[Limb]>(u2);
                // SAFETY: the complete u0 and live u2 inputs are disjoint;
                // add_sub permits its difference output to replace u2.
                unsafe {
                    SsaRing::add_sub(
                        u0,
                        difference,
                        difference.cast::<Limb>(),
                        mod_bits,
                        add_sub_kernel,
                    );
                }
            } else {
                // SAFETY: every butterfly call supplies complete disjoint
                // cl-limb slots, and this branch copies the live u0 input.
                unsafe {
                    copy_nonoverlapping(u0.as_ptr(), u2.as_mut_ptr(), cl);
                }
            }
            if has_q3 {
                let difference = from_mut::<[Limb]>(u3);
                // SAFETY: the complete u1 and live u3 inputs are disjoint,
                // with the difference replacing its consumed u3 input.
                unsafe {
                    SsaRing::add_sub(
                        u1,
                        difference,
                        difference.cast::<Limb>(),
                        mod_bits,
                        add_sub_kernel,
                    );
                }
            } else {
                // SAFETY: the common butterfly runs only with a live u1
                // input; its cl initialized limbs are disjoint from u3.
                unsafe {
                    copy_nonoverlapping(u1.as_ptr(), u3.as_mut_ptr(), cl);
                }
            }
            let sum_difference = from_mut::<[Limb]>(u1);
            // SAFETY: the two pair sums occupy complete disjoint slots.
            unsafe {
                SsaRing::add_sub(
                    u0,
                    sum_difference,
                    sum_difference.cast::<Limb>(),
                    mod_bits,
                    add_sub_kernel,
                );
            }
            let rotated_difference = from_mut::<[Limb]>(u3);
            // SAFETY: i_shift=n/2 is positive and reduced. Staging is a
            // complete private slot disjoint from both pair differences.
            unsafe {
                SsaRing::shift_from(staging, u3, i_shift, mod_bits);
                SsaRing::add_sub(
                    u2,
                    rotated_difference,
                    staging.as_ptr(),
                    mod_bits,
                    add_sub_kernel,
                );
            }
        };
        // active_len>q proves q0 and q1 both have a live first input. The
        // remaining pair inputs depend on the same support bounds as below.
        // SAFETY: q is positive, so every quarter has a complete first slot;
        // these initialized spans and staging are pairwise disjoint.
        unsafe {
            butterfly(
                q0.get_unchecked_mut(..cl),
                q1.get_unchecked_mut(..cl),
                q2.get_unchecked_mut(..cl),
                q3.get_unchecked_mut(..cl),
                scratch_slot,
                q2_start < active_len,
                q3_start < active_len,
            );
        }
        let mut twiddle_shift = root_shift;
        for i in 1..quarter_len {
            // SAFETY: the admitted four quarters each contain quarter_len
            // complete cl-limb coefficients; i addresses one disjoint quartet.
            let (u0, u1, u2, u3) = unsafe {
                let offset = i.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                (
                    q0.get_unchecked_mut(offset..end),
                    q1.get_unchecked_mut(offset..end),
                    q2.get_unchecked_mut(offset..end),
                    q3.get_unchecked_mut(offset..end),
                )
            };
            // SAFETY: i<q and each offset lies below the complete 4*q length.
            let (is_q1_active, is_q2_active, is_q3_active) = unsafe {
                (
                    i.unchecked_add(q) < active_len,
                    i.unchecked_add(q2_start) < active_len,
                    i.unchecked_add(q3_start) < active_len,
                )
            };
            // The earlier active_len <= quarter_len branch returns, so q0 is dense.

            if !is_q1_active {
                // u1 = u2 = u3 = 0 -> direct twiddle copy from u0
                let w1 = twiddle_shift;
                // SAFETY: this quarter's principal twiddle is below period/4,
                // so its double and triple are exact reduced exponents.
                let (w2, w3) = unsafe { (w1.unchecked_mul(2), w1.unchecked_mul(3)) };

                // active_len>q makes q1's first slot active. Its absence at
                // this index therefore proves i>0 and removes the identity
                // case from this sparse tail as well.
                // SAFETY: i>0 gives positive reduced w1, w2, and w3; the four
                // complete coefficient slots are pairwise disjoint.
                unsafe {
                    SsaRing::shift_from(u2, u0, w1, mod_bits);
                    SsaRing::shift_from(u1, u0, w2, mod_bits);
                    SsaRing::shift_from(u3, u0, w3, mod_bits);
                }
                // SAFETY: advancing one quarter index reaches at most period/4.
                twiddle_shift = unsafe { twiddle_shift.unchecked_add(root_shift) };
                continue;
            }

            butterfly(u0, u1, u2, u3, scratch_slot, is_q2_active, is_q3_active);
            // SAFETY: i>=1 makes w1, 2*w1 and 3*w1 positive. Since i<q,
            // all are below the admitted period. Complete output coefficients
            // and the private staging slot are initialized and disjoint.
            unsafe {
                let w2 = twiddle_shift.unchecked_mul(2);
                let w3 = twiddle_shift.unchecked_mul(3);
                SsaRing::shift_in_place(u1, w2, mod_bits, scratch_slot);
                SsaRing::shift_in_place(u2, twiddle_shift, mod_bits, scratch_slot);
                SsaRing::shift_in_place(u3, w3, mod_bits, scratch_slot);
            }

            // SAFETY: advancing one quarter index reaches at most period/4.
            twiddle_shift = unsafe { twiddle_shift.unchecked_add(root_shift) };
        }
    }

    /// Recursively parallelizes the dense radix-4 DIF stage across available threads.
    ///
    /// # Safety
    /// All four pairwise-disjoint quarter slices contain `count` complete slots
    /// for `ctx.mod_bits`; the parent partition establishes count's positivity.
    /// `scratch` has at least `coeff_limbs(mod_bits)` limbs.
    /// `start_twiddle + count*root_shift <= mod_bits/2` is the principal
    /// quarter range; `i_shift=mod_bits/2`.
    pub unsafe fn dif_radix4_dense_parallel<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        count: NonZeroUsize,
        start_twiddle: usize,
        root_shift: usize,
        i_shift: usize,
        scratch: &mut [Limb],
        ctx: &TransformContext<'_, E>,
    ) {
        let [q0, q1, q2, q3] = quarters;
        let cl = ctx.cl.get();
        let executor = ctx.executor;
        if count.get() >= 2
            && Self::should_parallelize(
                count.get(),
                cl,
                cl,
                scratch.len(),
                executor.parallelism().get(),
            )
        {
            let half = count.get() >> 1;
            // SAFETY: count>=2 gives 1<=half<count, so both children contain
            // positive complete coefficient ranges. Retain their widths once
            // at this partition boundary and propagate them to the leaves.
            let (left_count, right_count) = unsafe {
                (
                    NonZeroUsize::new_unchecked(half),
                    NonZeroUsize::new_unchecked(count.get().unchecked_sub(half)),
                )
            };
            // SAFETY: count complete coefficients exist in each quarter.
            let half_matrix = unsafe { half.unchecked_mul(cl) };
            // SAFETY: half_matrix <= q0.len() since half = count / 2.
            let (q0_left, q0_right) = unsafe { q0.split_at_mut_unchecked(half_matrix) };
            // SAFETY: half_matrix <= q1.len() since all quarters have identical length.
            let (q1_left, q1_right) = unsafe { q1.split_at_mut_unchecked(half_matrix) };
            // SAFETY: half_matrix <= q2.len() since all quarters have identical length.
            let (q2_left, q2_right) = unsafe { q2.split_at_mut_unchecked(half_matrix) };
            // SAFETY: half_matrix <= q3.len() since all quarters have identical length.
            let (q3_left, q3_right) = unsafe { q3.split_at_mut_unchecked(half_matrix) };

            let split_scratch = scratch.len().div_euclid(2);
            // SAFETY: should_parallelize proved scratch.len() >= 2 * cl, so both halves have >= cl.
            let (scratch_left, scratch_right) =
                unsafe { scratch.split_at_mut_unchecked(split_scratch) };

            // SAFETY: half<count, so this child start is below the parent's
            // quarter endpoint <=mod_bits/2 and needs no exponent reduction.
            let right_twiddle =
                unsafe { start_twiddle.unchecked_add(root_shift.unchecked_mul(half)) };

            let ((), ()) = executor.join(
                // SAFETY: left half matrices and left scratch are disjoint from right half.
                || unsafe {
                    Self::dif_radix4_dense_parallel(
                        [q0_left, q1_left, q2_left, q3_left],
                        left_count,
                        start_twiddle,
                        root_shift,
                        i_shift,
                        scratch_left,
                        ctx,
                    );
                },
                // SAFETY: right half matrices and right scratch are disjoint from left half.
                || unsafe {
                    Self::dif_radix4_dense_parallel(
                        [q0_right, q1_right, q2_right, q3_right],
                        right_count,
                        right_twiddle,
                        root_shift,
                        i_shift,
                        scratch_right,
                        ctx,
                    );
                },
            );
        } else {
            // SAFETY: quarters are disjoint and scratch holds at least cl limbs.
            unsafe {
                Self::dif_radix4_dense_block(
                    [q0, q1, q2, q3],
                    count,
                    start_twiddle,
                    root_shift,
                    i_shift,
                    scratch,
                    ctx,
                );
            }
        }
    }

    /// Computes a streaming dense radix-4 DIF butterfly pass.
    ///
    /// # Safety
    /// All four pairwise-disjoint quarter slices contain exactly `count`
    /// initialized complete slots for `ctx.mod_bits`.
    /// `scratch` has at least `coeff_limbs(mod_bits)` limbs.
    /// `start_twiddle + count*root_shift <= ctx.mod_bits/2` and
    /// `i_shift=ctx.mod_bits/2`; the primitive `root_shift` is positive.
    pub unsafe fn dif_radix4_dense_block<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        count: NonZeroUsize,
        start_twiddle: usize,
        root_shift: usize,
        i_shift: usize,
        scratch: &mut [Limb],
        ctx: &TransformContext<'_, E>,
    ) {
        let [q0, q1, q2, q3] = quarters;
        let cl = ctx.cl.get();
        let mod_bits = ctx.mod_bits;
        let add_sub_kernel = ctx.kernel;
        let mut twiddle_shift = start_twiddle;
        // SAFETY: the caller proves scratch holds at least cl limbs.
        let scratch_slot = unsafe { scratch.get_unchecked_mut(..cl) };

        // The butterfly's four disjoint coefficients have one shared ring.
        // Factoring its arithmetic from the twiddle schedule allows the single
        // identity to be peeled without duplicating any add/subtract kernel.
        let butterfly = move |u0: &mut [Limb],
                              u1: &mut [Limb],
                              u2: &mut [Limb],
                              u3: &mut [Limb],
                              staging: &mut [Limb]| {
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
            // Stage 3: Combine (u2, u3) with i_shift -> v2 in u2, v3 in u3
            // SAFETY: the admitted ring has i_shift=n/2>0, and the complete
            // coefficient and staging slot are disjoint.
            unsafe {
                SsaRing::shift_from(staging, u3, i_shift, mod_bits);
            }
            let u3_operand_ptr = staging.as_ptr();

            let u3_diff_dest = from_mut::<[Limb]>(u3);
            // SAFETY: u2, u3, u3_operand_ptr are disjoint cl-limb spans.
            unsafe {
                SsaRing::add_sub(u2, u3_diff_dest, u3_operand_ptr, mod_bits, add_sub_kernel);
            }
        };

        let start = if start_twiddle == 0 {
            // SAFETY: count is positive and each quarter holds count complete
            // initialized coefficients, so all four first slots exist.
            let [u0, u1, u2, u3] = unsafe {
                [
                    q0.get_unchecked_mut(..cl),
                    q1.get_unchecked_mut(..cl),
                    q2.get_unchecked_mut(..cl),
                    q3.get_unchecked_mut(..cl),
                ]
            };
            butterfly(u0, u1, u2, u3, scratch_slot);
            twiddle_shift = root_shift;
            1
        } else {
            0
        };

        for index in start..count.get() {
            // SAFETY: index<count and the four quarters each contain exactly
            // count complete cl-limb slots. Their owners and staging are disjoint.
            let [u0, u1, u2, u3] = unsafe {
                let offset = index.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                [
                    q0.get_unchecked_mut(offset..end),
                    q1.get_unchecked_mut(offset..end),
                    q2.get_unchecked_mut(offset..end),
                    q3.get_unchecked_mut(offset..end),
                ]
            };
            butterfly(u0, u1, u2, u3, scratch_slot);
            // SAFETY: the zero-index slot was peeled, or this worker's range
            // starts at a positive parent index. Thus 0<w1<period/4 and its
            // exact double and triple are positive reduced exponents. The
            // independent outputs and staging contain complete coefficients.
            unsafe {
                let w2 = twiddle_shift.unchecked_mul(2);
                let w3 = twiddle_shift.unchecked_mul(3);
                SsaRing::shift_in_place(u1, w2, mod_bits, scratch_slot);
                SsaRing::shift_in_place(u2, twiddle_shift, mod_bits, scratch_slot);
                SsaRing::shift_in_place(u3, w3, mod_bits, scratch_slot);
            }

            // SAFETY: the inclusive endpoint is at most the principal period/4.
            twiddle_shift = unsafe { twiddle_shift.unchecked_add(root_shift) };
        }
    }
}
