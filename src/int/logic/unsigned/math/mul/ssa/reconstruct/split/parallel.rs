//! Independent coefficient-range extraction with private twist staging.

#![expect(
    unsafe_code,
    reason = "Checked matrix and scratch layouts establish disjoint coefficient ranges with private staging buffers"
)]

use core::{num::NonZeroUsize, ops::Div};

use crate::parallel::ParallelExecutor;

use super::{
    LIMB_BITS, Limb, RingPeriods, SSA_PARALLEL_MIN_LIMB_WORK, SplitLayout, SsaCoefficients, SsaRing,
};

struct SplitWork {
    layout: SplitLayout,
    bits: usize,
    step: usize,
    root: usize,
    periods: RingPeriods,
}

impl SsaCoefficients {
    /// Splits and twists independent coefficient ranges using existing scratch.
    ///
    /// # Safety
    /// The dimensions, complete matrix and two-coefficient scratch satisfy
    /// `split_twisted`'s contract. The limb-aligned inner width has representable
    /// `4*inner_bits`. Each concurrent range owns two staging coefficients.
    #[expect(
        clippy::too_many_arguments,
        reason = "Parallel splitting carries the executor alongside the complete geometry"
    )]
    pub unsafe fn split_twisted_with_executor<E: ParallelExecutor>(
        src: &[Limb],
        matrix: &mut [Limb],
        count: usize,
        chunk: NonZeroUsize,
        cl: NonZeroUsize,
        periods: RingPeriods,
        step: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if executor.parallelism().get() == 1 {
            // SAFETY: identical dimensions and complete disjoint buffers.
            unsafe {
                Self::split_twisted(src, matrix, count, chunk, cl, periods, step, scratch);
            }
            return;
        }
        let work = SplitWork::new(chunk, cl, periods, step, 0);
        // SAFETY: split_twisted's contract supplies a representable source bit
        // capacity and positive chunk width from the admitted geometry.
        let active = unsafe { src.len().unchecked_mul(LIMB_BITS) }
            .div_ceil(chunk.get())
            .min(count);
        // SAFETY: active <= count and the complete count*cl matrix is representable.
        let span = unsafe { active.unchecked_mul(cl.get()) };
        // SAFETY: span<=count*cl<=matrix.len() by the complete matrix contract.
        let (prefix, zero_tail) = unsafe { matrix.split_at_mut_unchecked(span) };
        zero_tail.fill(0);
        // SAFETY: only prefix is active; the unpaired high slice is unused.
        unsafe {
            work.run::<false, E>(src, prefix, &mut [], 0, [0, 0], executor, scratch);
        }
    }

    /// Splits and twists both halves of the zero-high first DIF stage in parallel.
    ///
    /// # Safety
    /// The source's active chunks fit the lower half. Dimensions and complete
    /// disjoint matrix/scratch satisfy `split_twisted_and_stage1_dif`'s contract.
    #[expect(
        clippy::too_many_arguments,
        reason = "Fused parallel splitting binds both twist roots and the executor"
    )]
    pub unsafe fn split_twisted_and_stage1_dif_with_executor<E: ParallelExecutor>(
        src: &[Limb],
        matrix: &mut [Limb],
        count: usize,
        chunk: NonZeroUsize,
        cl: NonZeroUsize,
        periods: RingPeriods,
        step: usize,
        root: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) -> bool {
        if executor.parallelism().get() == 1 {
            // SAFETY: identical dimensions and complete disjoint buffers.
            return unsafe {
                Self::split_twisted_and_stage1_dif(
                    src, matrix, count, chunk, cl, periods, step, root, scratch,
                )
            };
        }
        if count < 2 {
            return false;
        }
        let work = SplitWork::new(chunk, cl, periods, step, root);
        let half = count >> 1;
        // SAFETY: split_twisted's contract supplies a representable source bit
        // capacity and positive chunk width from the admitted geometry.
        let active = unsafe { src.len().unchecked_mul(LIMB_BITS) }
            .div_ceil(chunk.get())
            .min(half);
        // SAFETY: these products are bounded by the checked complete matrix span.
        let (half_span, active_span) =
            unsafe { (half.unchecked_mul(cl.get()), active.unchecked_mul(cl.get())) };
        // SAFETY: half_span<=count*cl<=matrix.len(); active_span<=half_span,
        // and both exact half matrices hold the complete active prefix.
        let ((low_prefix, low_zero), (high_prefix, high_zero)) = unsafe {
            let (low, high) = matrix.split_at_mut_unchecked(half_span);
            (
                low.split_at_mut_unchecked(active_span),
                high.split_at_mut_unchecked(active_span),
            )
        };
        low_zero.fill(0);
        high_zero.fill(0);
        // SAFETY: aligned equal active halves are disjoint; all other slots are zero.
        unsafe {
            work.run::<true, E>(src, low_prefix, high_prefix, 0, [0, 0], executor, scratch);
        }
        true
    }
}

impl SplitWork {
    fn new(
        chunk: NonZeroUsize,
        cl: NonZeroUsize,
        periods: RingPeriods,
        step: usize,
        root: usize,
    ) -> Self {
        let bits = periods.whole.get() >> 1;
        let layout = SplitLayout::new(chunk, bits, cl);
        Self {
            layout,
            bits,
            step: SsaRing::reduce_mod_period(step, periods.half),
            root: SsaRing::reduce_mod_period(root, periods.whole),
            periods,
        }
    }

    /// # Safety
    /// Low contains complete active slots starting at absolute source chunk first;
    /// high has the identical span when PAIRED. Roots are reduced and scratch
    /// owns two complete coefficients. Every range remains in the outer geometry.
    #[expect(
        clippy::too_many_arguments,
        reason = "A split range retains absolute chunk and twist origins with private scratch"
    )]
    unsafe fn run<const PAIRED: bool, E: ParallelExecutor>(
        &self,
        src: &[Limb],
        low: &mut [Limb],
        high: &mut [Limb],
        first: usize,
        shifts: [usize; 2],
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let cl = self.layout.coefficient_len();
        let count = low.len().div(cl);
        let half = count >> 1;
        // SAFETY: low holds count complete coefficients, so half*cl<=low.len().
        // The admitted ring's representable 4*bits also bounds two guarded slots.
        let (partition_span, two_slots) =
            unsafe { (half.unchecked_mul(cl.get()), cl.get().unchecked_mul(2)) };
        if executor.parallelism().get() > 1
            && count >= 2
            && partition_span >= SSA_PARALLEL_MIN_LIMB_WORK
            && scratch.len() >> 1 >= two_slots
        {
            // SAFETY: partition_span is within the complete low range and, when PAIRED,
            // its equal high range. The unpaired high partition uses zero.
            let ((low_a, low_b), (high_a, high_b), (work_a, work_b)) = unsafe {
                (
                    low.split_at_mut_unchecked(partition_span),
                    high.split_at_mut_unchecked(if PAIRED { partition_span } else { 0 }),
                    scratch.split_at_mut_unchecked(scratch.len() >> 1),
                )
            };
            let right_shifts = [
                advance(shifts[0], self.step, half, self.periods.half),
                advance(shifts[1], self.root, half, self.periods.whole),
            ];
            // SAFETY: first+half stays inside the validated source chunk range.
            let right_first = unsafe { first.unchecked_add(half) };
            let ((), ()) = executor.join(
                // SAFETY: complete left ranges and two-slot arena are disjoint from right.
                || unsafe {
                    self.run::<PAIRED, E>(src, low_a, high_a, first, shifts, executor, work_a);
                },
                // SAFETY: right ranges carry absolute source indices and matching roots.
                || unsafe {
                    self.run::<PAIRED, E>(
                        src,
                        low_b,
                        high_b,
                        right_first,
                        right_shifts,
                        executor,
                        work_b,
                    );
                },
            );
            return;
        }
        // SAFETY: the range owns two complete private staging coefficients.
        let (stage, factor) = unsafe { scratch.split_at_mut_unchecked(cl.get()) };
        // SAFETY: the exact stage matches the validated chunk layout.
        unsafe {
            self.layout.initialize_padding(stage);
        }
        let [mut twist, mut root] = shifts;
        let mut high_slots = high.chunks_exact_mut(cl.get());
        for (offset, low_slot) in low.chunks_exact_mut(cl.get()).enumerate() {
            // SAFETY: first+offset is in the validated active source range;
            // stage's suffix remains zero because twists only read it.
            unsafe {
                SsaCoefficients::extract_chunk(
                    src,
                    stage,
                    first.unchecked_add(offset),
                    self.layout,
                );
            }
            // SAFETY: complete disjoint stage, output and factor slots; both
            // half-bit branches have a reduced whole shift and initialized input.
            unsafe {
                if twist & 1 == 0 {
                    SsaRing::shift_from(low_slot, stage, twist >> 1, self.bits);
                } else {
                    SsaRing::shift_sqrt2_from(low_slot, stage, twist >> 1, self.bits, factor);
                }
                if PAIRED {
                    // Equal low/high lengths provide one output for every iteration.
                    let high_slot = high_slots.next().unwrap_unchecked();
                    SsaRing::shift_from(high_slot, low_slot, root, self.bits);
                }
            }
            twist = SplitLayout::add_reduced(twist, self.step, self.periods.half);
            if PAIRED {
                root = SplitLayout::add_reduced(root, self.root, self.periods.whole);
            }
        }
    }
}

/// Computes a range's absolute reduced root without overflowing pointer width.
fn advance(start: usize, step: usize, count: usize, period: NonZeroUsize) -> usize {
    if let Some(sum) = step
        .checked_mul(count)
        .and_then(|product| start.checked_add(product))
    {
        return SsaRing::reduce_mod_period(sum, period);
    }
    let mut result = start;
    let mut remaining = count;
    let mut power = step;
    while remaining != 0 {
        if remaining & 1 != 0 {
            result = SplitLayout::add_reduced(result, power, period);
        }
        remaining >>= 1;
        power = SplitLayout::add_reduced(power, power, period);
    }
    result
}
