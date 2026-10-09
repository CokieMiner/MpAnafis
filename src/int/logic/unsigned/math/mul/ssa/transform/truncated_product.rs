//! Zero-tail polynomial products using only their required frequency prefixes.
//!
//! An operand pair with supports a,b has product support a+b-1. Truncation is
//! legal only below the full ring transform length: a wrapped negacyclic
//! convolution cannot supply the zero time tail required by the ITFT.

#![expect(
    unsafe_code,
    reason = "Operation-sized plans reserve complete input matrices and output chains; support proofs bound shortened inverse prefixes"
)]

use core::{array, mem::take, num::NonZeroUsize, ops::Div};

use crate::parallel::ParallelExecutor;

use super::{FftPlan, Limb, LimbOutput, SsaCoefficients, SsaTransform, TruncatedTransform};

impl SsaTransform {
    /// Runs TFTs, a prefix-only pointwise callback, then matching ITFTs and
    /// reconstruction. The first OUTPUTS matrices hold the product spectra.
    /// Shared products may request different frequency counts for each output.
    ///
    /// # Safety
    /// Each input has its declared active support under plan. Each shortened
    /// output prefix bounds its polynomial support; a full K-frequency prefix
    /// permits arbitrary wrapped ring products. Spectrum requests cover all
    /// callback reads, and all buffers are disjoint. The arities describe one
    /// multiplication (2,1), square (1,1), or shared product (3,2).
    /// Scratch has the complete plan-sized layout for this operation/executor.
    /// The callback multiplies exactly the supplied prefixes in the plan's ring.
    #[expect(
        clippy::too_many_arguments,
        reason = "Truncated product boundary binds operand, spectral, and output supports to the prepared plan"
    )]
    pub unsafe fn truncated_product<const INPUTS: usize, const OUTPUTS: usize, E, P>(
        inputs: [&[Limb]; INPUTS],
        mut outputs: [&mut [impl LimbOutput]; OUTPUTS],
        active: [usize; INPUTS],
        spectra: [usize; INPUTS],
        counts: [usize; OUTPUTS],
        plan: &FftPlan,
        executor: &E,
        scratch: &mut [Limb],
        pointwise: P,
    ) where
        E: ParallelExecutor,
        P: FnOnce([&mut [Limb]; INPUTS], &mut [Limb]),
    {
        // SAFETY: the operation's successful scratch plan contains INPUTS full
        // matrices, so this prefix product is representable on every target.
        let matrix_len = unsafe { plan.mat_limbs.get().unchecked_mul(INPUTS) };
        // SAFETY: the operation-sized arena reserves INPUTS complete matrices
        // followed by its disjoint staging and pointwise workspace.
        let (matrices, work) = unsafe { scratch.split_at_mut_unchecked(matrix_len) };
        // The constructed plan proves positive limb-aligned bits, representable
        // 4*bits, matching coefficient width, and primitive transform roots.
        // Architecture selection is retained once for all TFT/ITFT phases.
        let transform = TruncatedTransform::new(plan);
        // SAFETY: each matrix is a complete disjoint planned input; staging
        // initializes only its active prefix, which the TFT zero-extends.
        unsafe {
            Self::stage_many(
                matrices, &inputs, &active, &spectra, plan, &transform, executor, work,
            );
        }
        let mut remaining = matrices;
        let mut requests = spectra.into_iter();
        let prefixes = array::from_fn(|_| {
            // SAFETY: INPUTS complete spans are consumed exactly INPUTS times.
            let (matrix, tail) =
                unsafe { take(&mut remaining).split_at_mut_unchecked(plan.mat_limbs.get()) };
            remaining = tail;
            // Array lengths coincide, so one request exists per complete matrix.
            // SAFETY: INPUTS requests are consumed exactly INPUTS times.
            let count = unsafe { requests.next().unwrap_unchecked() };
            // SAFETY: count <= K and K*inner_cl is the representable matrix span.
            let prefix = unsafe { count.unchecked_mul(plan.inner_cl.get()) };
            // SAFETY: count<=K, so its complete frequency prefix fits matrix.
            unsafe { matrix.get_unchecked_mut(..prefix) }
        });
        pointwise(prefixes, work);
        // Dead input matrices join the arena only after the pointwise callback
        // returns. No extra full-matrix snapshot or spectrum padding is needed.
        // SAFETY: OUTPUTS <= INPUTS, whose complete matrix product fits above.
        let output_len = unsafe { plan.mat_limbs.get().unchecked_mul(OUTPUTS) };
        // SAFETY: this prefix is no larger than the input matrix region. The
        // pointwise borrows have ended, releasing dead matrices into the arena.
        let (results, reconstruction) = unsafe { scratch.split_at_mut_unchecked(output_len) };
        // SAFETY: callback establishes exactly counts[i] product frequencies;
        // the zero-tail support proof permits each inverse's independent count.
        unsafe {
            Self::finish_many(
                results,
                &mut outputs,
                &counts,
                plan,
                &transform,
                executor,
                reconstruction,
            );
        }
    }

    /// Returns the polynomial support bound, capped at the ring length.
    /// An empty operand produces an empty polynomial, independently of its peer.
    pub fn product_support(left: usize, right: usize, len: usize) -> usize {
        if left == 0 || right == 0 {
            return 0;
        }
        // Overflow means the support exceeds every representable ring length;
        // this is a legal full-transform case, not an infallible unwrap.
        left.checked_add(right)
            .and_then(|sum| sum.checked_sub(1))
            .unwrap_or(len)
            .min(len)
    }
}

impl SsaTransform {
    /// Stages independent operands with private coefficient arenas at every fork.
    ///
    /// # Safety
    /// Inputs, supports, and spectra have equal lengths. Matrices contain that many
    /// full plan spans. Scratch provides at least two coefficients per active fork.
    #[expect(
        clippy::too_many_arguments,
        reason = "Operand staging carries the shared transform context and input/output support slices"
    )]
    unsafe fn stage_many<E: ParallelExecutor>(
        matrices: &mut [Limb],
        inputs: &[&[Limb]],
        active: &[usize],
        spectra: &[usize],
        plan: &FftPlan,
        transform: &TruncatedTransform,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        // SAFETY: representable 4*inner_bits and LIMB_BITS >= 16 prove
        // 4*(inner_bits/LIMB_BITS+1) fits on every supported pointer width.
        let minimum_fork = unsafe { plan.inner_cl.get().unchecked_mul(4) };
        if inputs.len() > 1 && executor.parallelism().get() > 1 && scratch.len() >= minimum_fork {
            let half = inputs.len() >> 1;
            // SAFETY: half is at most the complete matrix count in this arena.
            let offset = unsafe { half.unchecked_mul(plan.mat_limbs.get()) };
            // SAFETY: all three metadata slices have inputs.len() entries and the
            // matrix arena has that many complete spans; half partitions each.
            let (
                (left, right),
                (left_inputs, right_inputs),
                (left_active, right_active),
                (left_spectra, right_spectra),
            ) = unsafe {
                (
                    matrices.split_at_mut_unchecked(offset),
                    inputs.split_at_unchecked(half),
                    active.split_at_unchecked(half),
                    spectra.split_at_unchecked(half),
                )
            };
            let (first, second) = scratch.split_at_mut(scratch.len() >> 1);
            let ((), ()) = executor.join(
                // SAFETY: complete disjoint matrix/scratch partitions and matching supports.
                || unsafe {
                    Self::stage_many(
                        left,
                        left_inputs,
                        left_active,
                        left_spectra,
                        plan,
                        transform,
                        executor,
                        first,
                    );
                },
                // SAFETY: complete disjoint matrix/scratch partitions and matching supports.
                || unsafe {
                    Self::stage_many(
                        right,
                        right_inputs,
                        right_active,
                        right_spectra,
                        plan,
                        transform,
                        executor,
                        second,
                    );
                },
            );
            return;
        }
        for (((matrix, &input), &support), &count) in matrices
            .chunks_exact_mut(plan.mat_limbs.get())
            .zip(inputs)
            .zip(active)
            .zip(spectra)
        {
            // SAFETY: support <= K and K*inner_cl is the representable matrix span.
            let active_len = unsafe { support.unchecked_mul(plan.inner_cl.get()) };
            // SAFETY: support<=K bounds this initialized prefix by the matrix span.
            let active_matrix = unsafe { matrix.get_unchecked_mut(..active_len) };
            // SAFETY: support bounds the source polynomial; split only writes that
            // complete prefix and the TFT treats the rest as implicit zero. Both
            // share plan roots, ring width, and a disjoint two-coefficient arena.
            unsafe {
                SsaCoefficients::split_twisted_with_executor(
                    input,
                    active_matrix,
                    support,
                    plan.chunk_bits,
                    plan.inner_cl,
                    plan.periods,
                    plan.twist_step_half,
                    executor,
                    scratch,
                );
                transform.forward(
                    matrix,
                    plan.transform_len,
                    plan.twist_step_half,
                    support,
                    count,
                    executor,
                    scratch,
                );
            }
        }
    }

    /// Inverts and reconstructs independent output matrices with separate counts.
    ///
    /// # Safety
    /// Matrices hold one full plan span per output, with `counts[i]` valid frequencies
    /// of a polynomial zero above `counts[i]`. Scratch covers simultaneous chains as
    /// reserved by the multiply, square, or shared-product planner.
    unsafe fn finish_many<E: ParallelExecutor>(
        matrices: &mut [Limb],
        outputs: &mut [&mut [impl LimbOutput]],
        counts: &[usize],
        plan: &FftPlan,
        transform: &TruncatedTransform,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        // SAFETY: the successful executor-sized plan reserves this exact twiddle
        // product for each concurrent output chain before execution begins.
        let twiddle_len = unsafe {
            plan.inner_cl
                .get()
                .unchecked_mul(plan.parallel_slots(executor.parallelism().get()))
        };
        if outputs.len() > 1 && executor.parallelism().get() > 1 {
            // SAFETY: the pair plan explicitly reserves two simultaneous complete
            // reconstruction chains, each containing this sum of private arenas.
            let chain_len = unsafe {
                twiddle_len.unchecked_add(plan.reconstruction_scratch(executor.parallelism().get()))
            };
            // SAFETY: this is the two-output shared-product case; each metadata
            // slice has two entries, and both complete chains were reserved.
            let (
                (left, right),
                (first_output, second_output),
                (first_count, second_count),
                (first, second),
            ) = unsafe {
                (
                    matrices.split_at_mut_unchecked(plan.mat_limbs.get()),
                    outputs.split_at_mut_unchecked(1),
                    counts.split_at_unchecked(1),
                    scratch.split_at_mut_unchecked(chain_len),
                )
            };
            let ((), ()) = executor.join(
                // SAFETY: first complete output chain owns its planned private arena.
                || unsafe {
                    Self::finish_many(
                        left,
                        first_output,
                        first_count,
                        plan,
                        transform,
                        executor,
                        first,
                    );
                },
                // SAFETY: second complete chain owns the remaining private arena.
                || unsafe {
                    Self::finish_many(
                        right,
                        second_output,
                        second_count,
                        plan,
                        transform,
                        executor,
                        second,
                    );
                },
            );
            return;
        }
        // SAFETY: the complete output chain reserves twiddle_len limbs before
        // reconstruction; both partitions are exclusively owned and initialized.
        let (twiddle, reconstruction) = unsafe { scratch.split_at_mut_unchecked(twiddle_len) };
        for ((matrix, output), &count) in matrices
            .chunks_exact_mut(plan.mat_limbs.get())
            .zip(outputs)
            .zip(counts)
        {
            if count == 0 {
                output.fill(LimbOutput::from_limb(0));
                continue;
            }
            // For L = next_power_of_two(count) <= K and j < L, bitrev_K(j)
            // equals (K/L)*bitrev_L(j). Hence the first count frequencies are
            // identical under the primitive L-th root. The product's zero tail
            // permits an L-point inverse, avoiding log2(K/L) doubling passes.
            // SAFETY: 1 <= count <= K and K is a representable power of two;
            // L*inner_cl is bounded by the plan's complete matrix span.
            let (inverse_len, inverse_log, inverse_span) = unsafe {
                let log = usize::BITS.unchecked_sub(count.unchecked_sub(1).leading_zeros());
                let len = NonZeroUsize::new_unchecked(1_usize.unchecked_shl(log));
                (len, log, len.get().unchecked_mul(plan.inner_cl.get()))
            };
            // SAFETY: the rounded length is at most K and the complete matrix
            // contains K initialized slots, so this inverse prefix fits.
            let inverse_matrix = unsafe { matrix.get_unchecked_mut(..inverse_span) };
            let inverse_root = transform.period.get().div(inverse_len);
            let mut inverse_twist = plan.inverse_twist();
            #[expect(
                clippy::as_conversions,
                reason = "inverse_log < usize::BITS <= 64 and fits both SSA pointer widths"
            )]
            {
                inverse_twist.transform_log = inverse_log as usize;
            }
            // SAFETY: exactly count product frequencies and a proven zero time tail
            // establish count scaled coefficients. Reconstruction reads only these
            // coefficients and divides by inverse_len. The enclosing K-point ring,
            // coefficient twists, magnitude bound, and accumulator sizing stay fixed.
            unsafe {
                transform.inverse(
                    inverse_matrix,
                    inverse_len.get(),
                    inverse_root,
                    count,
                    count,
                    false,
                    executor,
                    twiddle,
                );
                // count <= K and K*inner_cl is the representable matrix span.
                let prefix_len = count.unchecked_mul(plan.inner_cl.get());
                let prefix = matrix.get_unchecked_mut(..prefix_len);
                SsaCoefficients::reconstruct(
                    prefix,
                    plan.transform_len,
                    plan.chunk_bits,
                    plan.inner_bits,
                    plan.modulus_bits,
                    output,
                    reconstruction,
                    Some((inverse_twist, twiddle)),
                    executor,
                );
            }
        }
    }
}
