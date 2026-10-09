//! Radix-4 decimation-in-time stages and streaming coefficient leaves.

#![expect(
    unsafe_code,
    reason = "FFT transform kernels use unchecked access only after validated matrix and scratch proofs"
)]

use core::{num::NonZeroUsize, ptr::from_mut};

use crate::parallel::ParallelExecutor;

use super::{Limb, SsaRing, SsaTransform, TransformContext};

impl SsaTransform {
    /// Computes a single radix-4 DIT pass across 4 disjoint quarters.
    ///
    /// At quarter index j, the principal inverse exponents are -j*root,
    /// -2j*root and -3j*root modulo 2n. For j>0, j*root<n/2: all three
    /// exponents are positive differences below 2n. Index zero skips the shifts.
    ///
    /// # Safety
    /// All four disjoint quarter slices contain `quarter_width` complete slots
    /// for `ctx.mod_bits`. `root_shift*(4*quarter_width)=2*ctx.mod_bits`;
    /// scratch contains a complete coefficient and `needed_out<=4*quarter_width`.
    #[expect(
        clippy::too_many_lines,
        reason = "Radix-4 DIT stage fuses two unrolled butterfly stages with strided coefficient prefetching"
    )]
    pub unsafe fn dit_radix4_stage<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        quarter_width: NonZeroUsize,
        root_shift: usize,
        scratch: &mut [Limb],
        needed_out: usize,
        ctx: &TransformContext<'_, E>,
    ) {
        let [q0, q1, q2, q3] = quarters;
        let cl = ctx.cl.get();
        let quarter_len = quarter_width.get();
        let period = ctx.period.get();
        let mod_bits = ctx.mod_bits;
        // SAFETY: the principal root is positive and at most period/4.
        let twiddle_step = unsafe { period.unchecked_sub(root_shift) };

        if needed_out <= quarter_len {
            if needed_out == 0 {
                return;
            }
            // The first inverse twiddle is the identity. Its sum-only
            // butterfly is evaluated once before the positive-exponent loop.
            // SAFETY: 0<needed_out<=quarter_len supplies four disjoint complete
            // first slots. Semi-normalized addition preserves their guard bound.
            unsafe {
                let first0 = q0.get_unchecked_mut(..cl);
                let first1 = q1.get_unchecked(..cl);
                let first2 = q2.get_unchecked_mut(..cl);
                let first3 = q3.get_unchecked(..cl);
                SsaRing::add_in_place(first0, first1, mod_bits);
                SsaRing::add_in_place(first2, first3, mod_bits);
                SsaRing::add_in_place(first0, first2, mod_bits);
            }
            let mut twiddle_shift = twiddle_step;
            // SAFETY: the caller contract requires at least one initialized cl-limb
            // scratch slot; the range is therefore in bounds and never empty.
            let staging = unsafe { scratch.get_unchecked_mut(..cl) };
            for i in 1..needed_out {
                // SAFETY: i < needed_out <= quarter_len ensures in-bounds for all 4 disjoint quarters.
                let (v0, v1, v2, v3) = unsafe {
                    let offset = i.unchecked_mul(cl);
                    let end = offset.unchecked_add(cl);
                    (
                        q0.get_unchecked_mut(offset..end),
                        q1.get_unchecked_mut(offset..end),
                        q2.get_unchecked_mut(offset..end),
                        q3.get_unchecked_mut(offset..end),
                    )
                };

                // SAFETY: j>=1 gives positive forward=j*root<period/4 and
                // positive reduced inverse twiddles. The three complete slots
                // are disjoint from staging. The next index<=quarter_len keeps
                // the descending inverse exponent >=3*period/4.
                unsafe {
                    let forward = period.unchecked_sub(twiddle_shift);
                    let w2 = period.unchecked_sub(forward.unchecked_mul(2));
                    let w3 = period.unchecked_sub(forward.unchecked_mul(3));
                    SsaRing::shift_in_place(v1, w2, mod_bits, staging);
                    SsaRing::shift_in_place(v2, twiddle_shift, mod_bits, staging);
                    SsaRing::shift_in_place(v3, w3, mod_bits, staging);
                    twiddle_shift = twiddle_shift.unchecked_sub(root_shift);
                }

                // Only u0 = (v0+v1')+(v2'+v3') is requested. All three
                // difference outputs lie outside the retained prefix.
                // SAFETY: the four complete cl-limb slots are disjoint and
                // semi-normalized; each addition preserves that guard bound.
                unsafe {
                    SsaRing::add_in_place(v0, v1, mod_bits);
                    SsaRing::add_in_place(v2, v3, mod_bits);
                    SsaRing::add_in_place(v0, v2, mod_bits);
                }
            }
            return;
        }

        // SAFETY: the imaginary-unit exponent n/2 is below the period 2n.
        let inv_i = unsafe { period.unchecked_sub(mod_bits >> 1) };

        // SAFETY: four complete quarters form the validated transform length.
        let complete_len = unsafe { quarter_len.unchecked_mul(4) };
        if needed_out >= complete_len {
            // SAFETY: quarters are disjoint, scratch has at least cl limbs.
            unsafe {
                Self::dit_radix4_dense_parallel(
                    [q0, q1, q2, q3],
                    quarter_width,
                    0,
                    twiddle_step,
                    inv_i,
                    scratch,
                    ctx,
                );
            }
            return;
        }

        // SAFETY: the caller contract requires at least one initialized cl-limb
        // scratch slot; the range is therefore in bounds and never empty.
        let staging = unsafe { scratch.get_unchecked_mut(..cl) };

        let q = quarter_len;
        // SAFETY: q<needed_out<=4*q, and the complete matrix span fits.
        // The second-quarter count is positive; both later quarter offsets fit.
        let (second_count, third_start, fourth_start) = unsafe {
            (
                needed_out.unchecked_sub(q),
                q.unchecked_mul(2),
                q.unchecked_mul(3),
            )
        };
        let third_count = needed_out.saturating_sub(third_start);
        let fourth_count = needed_out.saturating_sub(fourth_start);
        // needed_out>q establishes the first q0 and q1 outputs. At index
        // zero all three untwiddles are identities; evaluate its full
        // butterfly once before advancing to positive inverse exponents.
        // SAFETY: q is positive and all quarters contain a complete initialized
        // first coefficient disjoint from the exact staging slot.
        unsafe {
            let v0 = q0.get_unchecked_mut(..cl);
            let v1 = q1.get_unchecked_mut(..cl);
            let v2 = q2.get_unchecked_mut(..cl);
            let v3 = q3.get_unchecked_mut(..cl);
            if fourth_count != 0 {
                Self::dit_radix4_butterfly::<true, true, E>(v0, v1, v2, v3, staging, inv_i, ctx);
            } else if third_count != 0 {
                Self::dit_radix4_butterfly::<true, false, E>(v0, v1, v2, v3, staging, inv_i, ctx);
            } else {
                Self::dit_radix4_butterfly::<false, false, E>(v0, v1, v2, v3, staging, inv_i, ctx);
            }
        }
        let mut twiddle_shift = twiddle_step;
        for i in 1..quarter_len {
            // SAFETY: all four disjoint quarters contain quarter_len complete
            // initialized cl-limb coefficients, including the quartet at i.
            let (v0, v1, v2, v3) = unsafe {
                let offset = i.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                (
                    q0.get_unchecked_mut(offset..end),
                    q1.get_unchecked_mut(offset..end),
                    q2.get_unchecked_mut(offset..end),
                    q3.get_unchecked_mut(offset..end),
                )
            };
            // The prefix-only branch returns for needed_out <= quarter_len;
            // every q0 slot is therefore requested throughout this loop.

            // SAFETY: i>=1 gives 0<forward=i*root<period/4. The three
            // inverse exponents are therefore positive and reduced. Slots
            // and staging are complete and disjoint; advancing to i+1<=q
            // keeps the descending principal exponent at least 3*period/4.
            unsafe {
                let forward = period.unchecked_sub(twiddle_shift);
                let w2 = period.unchecked_sub(forward.unchecked_mul(2));
                let w3 = period.unchecked_sub(forward.unchecked_mul(3));
                SsaRing::shift_in_place(v1, w2, mod_bits, staging);
                SsaRing::shift_in_place(v2, twiddle_shift, mod_bits, staging);
                SsaRing::shift_in_place(v3, w3, mod_bits, staging);
                twiddle_shift = twiddle_shift.unchecked_sub(root_shift);
            }

            if i >= second_count {
                // Only v0 (q0) is needed: u0 = (v0 + v1') + (v2' + v3')
                // SAFETY: the four complete semi-normalized slots are disjoint;
                // the three discarded differences have no subsequent reader.
                unsafe {
                    SsaRing::add_in_place(v0, v1, mod_bits);
                    SsaRing::add_in_place(v2, v3, mod_bits);
                    SsaRing::add_in_place(v0, v2, mod_bits);
                }
                continue;
            }

            // SAFETY: this requested quartet has complete disjoint
            // semi-normalized inputs and private staging for i^-1. The
            // prefix counts select only the required third/fourth outputs.
            unsafe {
                if i < fourth_count {
                    Self::dit_radix4_butterfly::<true, true, E>(
                        v0, v1, v2, v3, staging, inv_i, ctx,
                    );
                } else if i < third_count {
                    Self::dit_radix4_butterfly::<true, false, E>(
                        v0, v1, v2, v3, staging, inv_i, ctx,
                    );
                } else {
                    Self::dit_radix4_butterfly::<false, false, E>(
                        v0, v1, v2, v3, staging, inv_i, ctx,
                    );
                }
            }
        }
    }

    /// Recursively parallelizes the dense radix-4 DIT stage across available threads.
    ///
    /// # Safety
    /// All four pairwise-disjoint quarter slices contain `count` complete slots
    /// for `ctx.mod_bits`; the parent partition establishes count's positivity.
    /// `scratch` has at least `coeff_limbs(mod_bits)` limbs.
    /// `positive_step=2*mod_bits-twiddle_step` is the principal root;
    /// every slot belongs to its first quarter. `inv_i=3*mod_bits/2`.
    pub unsafe fn dit_radix4_dense_parallel<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        count: NonZeroUsize,
        start_twiddle: usize,
        twiddle_step: usize,
        inv_i: usize,
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
            // SAFETY: count>=2 gives 1<=half<count. The partition proves both
            // positive complete child widths once; recursive leaves retain them.
            let (left_count, right_count) = unsafe {
                (
                    NonZeroUsize::new_unchecked(half),
                    NonZeroUsize::new_unchecked(count.get().unchecked_sub(half)),
                )
            };
            // SAFETY: half<=count and each quarter contains count complete slots.
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

            let period = ctx.period.get();
            // SAFETY: positive_step is the principal root and half is within
            // its quarter. Thus delta<period/4. Reduced subtraction expresses
            // the inverse advance without multiplying the large negative step.
            let right_twiddle = unsafe {
                let delta = period.unchecked_sub(twiddle_step).unchecked_mul(half);
                if start_twiddle >= delta {
                    start_twiddle.unchecked_sub(delta)
                } else {
                    period.unchecked_sub(delta.unchecked_sub(start_twiddle))
                }
            };

            let ((), ()) = executor.join(
                // SAFETY: left half matrices and left scratch are disjoint from right half.
                || unsafe {
                    Self::dit_radix4_dense_parallel(
                        [q0_left, q1_left, q2_left, q3_left],
                        left_count,
                        start_twiddle,
                        twiddle_step,
                        inv_i,
                        scratch_left,
                        ctx,
                    );
                },
                // SAFETY: right half matrices and right scratch are disjoint from left half.
                || unsafe {
                    Self::dit_radix4_dense_parallel(
                        [q0_right, q1_right, q2_right, q3_right],
                        right_count,
                        right_twiddle,
                        twiddle_step,
                        inv_i,
                        scratch_right,
                        ctx,
                    );
                },
            );
        } else {
            // SAFETY: quarters are disjoint and scratch holds at least cl limbs.
            unsafe {
                Self::dit_radix4_dense_block(
                    [q0, q1, q2, q3],
                    count,
                    start_twiddle,
                    twiddle_step,
                    inv_i,
                    scratch,
                    ctx,
                );
            }
        }
    }

    /// Computes a streaming dense radix-4 DIT butterfly pass.
    ///
    /// # Safety
    /// All four pairwise-disjoint quarter slices contain exactly `count`
    /// initialized complete slots for `ctx.mod_bits`.
    /// `scratch` has at least `coeff_limbs(mod_bits)` limbs.
    /// `2*mod_bits-twiddle_step` is the principal root of the parent transform.
    /// This range lies in its first quarter, and `inv_i=3*mod_bits/2`.
    pub unsafe fn dit_radix4_dense_block<E: ParallelExecutor>(
        quarters: [&mut [Limb]; 4],
        count: NonZeroUsize,
        start_twiddle: usize,
        twiddle_step: usize,
        inv_i: usize,
        scratch: &mut [Limb],
        ctx: &TransformContext<'_, E>,
    ) {
        let [q0, q1, q2, q3] = quarters;
        let cl = ctx.cl.get();
        let mod_bits = ctx.mod_bits;
        let period = ctx.period.get();
        // SAFETY: twiddle_step=period-root and the positive principal root
        // is bounded by period/4. Its exact difference is already admitted.
        let positive_step = unsafe { period.unchecked_sub(twiddle_step) };
        let mut twiddle_shift = start_twiddle;
        // SAFETY: the caller contract requires at least one initialized cl-limb
        // scratch slot; the range is therefore in bounds and never empty.
        let staging = unsafe { scratch.get_unchecked_mut(..cl) };

        let start = if start_twiddle == 0 {
            // SAFETY: positive count and the complete quarter-width contract
            // give four disjoint initialized first coefficients.
            let [v0, v1, v2, v3] = unsafe {
                [
                    q0.get_unchecked_mut(..cl),
                    q1.get_unchecked_mut(..cl),
                    q2.get_unchecked_mut(..cl),
                    q3.get_unchecked_mut(..cl),
                ]
            };
            // SAFETY: first slots and staging are complete and disjoint;
            // the shared context and i^-1 match their admitted ring.
            unsafe {
                Self::dit_radix4_butterfly::<true, true, E>(v0, v1, v2, v3, staging, inv_i, ctx);
            }
            twiddle_shift = twiddle_step;
            1
        } else {
            0
        };

        for index in start..count.get() {
            // SAFETY: index<count addresses one complete cl-limb coefficient
            // in every disjoint quarter; all offset products fit those spans.
            let [v0, v1, v2, v3] = unsafe {
                let offset = index.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                [
                    q0.get_unchecked_mut(offset..end),
                    q1.get_unchecked_mut(offset..end),
                    q2.get_unchecked_mut(offset..end),
                    q3.get_unchecked_mut(offset..end),
                ]
            };
            // SAFETY: the identity coefficient was peeled, or this worker's
            // start has a positive parent index. Thus 0<forward<period/4;
            // its double and triple stay below period. Every slot is complete
            // and disjoint from staging; the next inverse remains positive.
            unsafe {
                let forward = period.unchecked_sub(twiddle_shift);
                let w2 = period.unchecked_sub(forward.unchecked_mul(2));
                let w3 = period.unchecked_sub(forward.unchecked_mul(3));
                SsaRing::shift_in_place(v1, w2, mod_bits, staging);
                SsaRing::shift_in_place(v2, twiddle_shift, mod_bits, staging);
                SsaRing::shift_in_place(v3, w3, mod_bits, staging);
                twiddle_shift = twiddle_shift.unchecked_sub(positive_step);
            }
            // SAFETY: untwiddling preserves the complete semi-normalized
            // coefficients, which remain disjoint from private staging.
            unsafe {
                Self::dit_radix4_butterfly::<true, true, E>(v0, v1, v2, v3, staging, inv_i, ctx);
            }
        }
    }

    /// Combines four untwiddled coefficients using `i^-1=2^(3n/2)`.
    ///
    /// `a=(v0+v1, v0-v1)`, `b=(v2+v3, v2-v3)` give
    /// `(a0+b0, a1+i^-1*b1, a0-b0, a1-i^-1*b1)` in place.
    /// The same guarded butterflies serve partial and dense stages. A
    /// discarded even or odd difference uses addition alone, leaving its
    /// consumed second input without a final store or difference carry chain.
    ///
    /// # Safety
    /// The four coefficients and staging are initialized pairwise-disjoint
    /// complete `ctx.cl`-limb slots for the admitted ring; `inv_i=3n/2`.
    unsafe fn dit_radix4_butterfly<const EVEN_TAIL: bool, const ODD_TAIL: bool, E>(
        v0: &mut [Limb],
        v1: &mut [Limb],
        v2: &mut [Limb],
        v3: &mut [Limb],
        staging: &mut [Limb],
        inv_i: usize,
        ctx: &TransformContext<'_, E>,
    ) {
        let mod_bits = ctx.mod_bits;
        let kernel = ctx.kernel;
        let first_difference = from_mut::<[Limb]>(v1);
        let second_difference = from_mut::<[Limb]>(v3);
        // SAFETY: each sum/difference pair has disjoint complete inputs;
        // the difference replaces its consumed second input.
        unsafe {
            SsaRing::add_sub(
                v0,
                first_difference,
                first_difference.cast::<Limb>(),
                mod_bits,
                kernel,
            );
            SsaRing::add_sub(
                v2,
                second_difference,
                second_difference.cast::<Limb>(),
                mod_bits,
                kernel,
            );
        }
        // SAFETY: the two pair sums are complete disjoint inputs; this fresh
        // difference pointer replaces v2 after its preceding mutable borrow,
        // or only their sum is written when the even tail is discarded.
        unsafe {
            if EVEN_TAIL {
                let even_output = from_mut::<[Limb]>(v2);
                SsaRing::add_sub(
                    v0,
                    even_output,
                    even_output.cast::<Limb>(),
                    mod_bits,
                    kernel,
                );
            } else {
                SsaRing::add_in_place(v0, v2, mod_bits);
            }
        }
        // SAFETY: 0<3n/2<2n is a reduced exponent. Staging does not overlap
        // the odd difference or either complete final output coefficient.
        unsafe {
            SsaRing::shift_from(staging, v3, inv_i, mod_bits);
            if ODD_TAIL {
                let odd_output = from_mut::<[Limb]>(v3);
                SsaRing::add_sub(v1, odd_output, staging.as_ptr(), mod_bits, kernel);
            } else {
                SsaRing::add_in_place(v1, staging, mod_bits);
            }
        }
    }
}
