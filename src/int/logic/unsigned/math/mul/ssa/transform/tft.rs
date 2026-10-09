//! Prefix TFT in the bit-reversed order of the radix-four DIF kernels.
//!
//! Boundary nodes form only requested frequency children. Matrices above the
//! tuned cache budget use balanced row/column decomposition; complete resident
//! children use fused radix-four transforms. Both retain O(L + n log L)
//! butterflies and one scratch coefficient per concurrent branch. Disjoint
//! coefficient-column scheduling for the boundary passes lives here as well:
//! columns within one pass are independent and join before the next row
//! transform, and each fork owns one staging coefficient.
//!
//! # References
//!
//! - van der Hoeven, J. (2004). The Truncated Fourier Transform and Applications.
//!   *Proceedings of ISSAC '04*, 376–383. <https://doi.org/10.1145/1005285.1005338>
//! - Harvey, D. (2009). A cache-friendly truncated Fourier transform.
//!   *Journal of Symbolic Computation*, 44(5), 474–483.

#![expect(
    unsafe_code,
    reason = "Constructed roots and complete matrix partitions bound column slots; active supports prevent reads of implicit zero tails"
)]

use core::{
    mem::size_of,
    num::NonZeroUsize,
    ops::{Div, Range},
    ptr::from_mut,
};

use crate::parallel::ParallelExecutor;

use super::{
    ArchKernels, CACHE_BLOCK_BYTES, CoefficientView, FftPlan, Limb, SsaRing, SsaTransform,
    TransformContext,
};

type ButterflyKernel = unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb);

/// Ring dimensions and architecture selection retained across TFT/ITFT nodes.
pub struct TruncatedTransform {
    pub bits: usize,
    pub cl: NonZeroUsize,
    pub period: NonZeroUsize,
    pub kernel: ButterflyKernel,
    pub max_resident: usize,
}

impl TruncatedTransform {
    /// Builds a truncated transform context for the given FFT plan.
    pub fn new(plan: &FftPlan) -> Self {
        Self {
            bits: plan.inner_bits,
            cl: plan.inner_cl,
            period: plan.periods.whole,
            kernel: ArchKernels::selected_add_sub_from_limbs_unchecked(),
            max_resident: CACHE_BLOCK_BYTES
                .div_euclid(size_of::<Limb>())
                .div(plan.inner_cl),
        }
    }
    /// DIF recurrence: `u_i = a_i+a_(h+i), v_i = omega^i(a_i-a_(h+i))`.
    /// Missing input coefficients and unrequested frequency children are omitted.
    ///
    /// # Safety
    /// The positive limb-aligned ring has representable `4*bits`, `period=2*bits`,
    /// `cl=coeff_limbs(bits)`, and a kernel selected for the current architecture.
    /// `len` is a power of two with `root*len=period`. The matrix contains exactly
    /// len complete initialized slots; both supports are <= len. Only the first
    /// inputs slots must be semi-normalized; the physical tail is arbitrary.
    /// Scratch contains a disjoint complete coefficient. The constructed FFT
    /// plan establishes these invariants before this infallible execution.
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "TFT decomposition schedules inputs, outputs, matrix kernels, and recursive DIF delegates"
    )]
    pub unsafe fn forward<E: ParallelExecutor>(
        &self,
        matrix: &mut [Limb],
        len: usize,
        root: usize,
        inputs: usize,
        outputs: usize,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if outputs == 0 {
            return;
        }
        if inputs == 0 {
            // SAFETY: outputs <= len and matrix contains len complete slots;
            // these are the only frequencies read by the pointwise product.
            unsafe {
                let span = outputs.unchecked_mul(self.cl.get());
                matrix.get_unchecked_mut(..span).fill(0);
            }
            return;
        }
        if len == 1 {
            return;
        }
        if len >= 4 && len > self.max_resident {
            // SAFETY: the complete matrix owns len slots and carries the
            // nonempty validated input/output supports and private scratch.
            unsafe {
                self.matrix_forward(
                    CoefficientView::new(matrix, len, self.cl),
                    root,
                    0,
                    inputs,
                    outputs,
                    executor,
                    scratch,
                );
            }
            return;
        }
        if outputs == len {
            // Sparse DIF reads only its active prefix. Each radix-four stage
            // establishes min(inputs, len/4) slots in every child, and the
            // two-point leaf copies a lone input without reading its peer.
            // No physical zero tail is needed for this nonempty transform.
            let ctx = TransformContext {
                mod_bits: self.bits,
                cl: self.cl,
                period: self.period,
                kernel: self.kernel,
                executor,
            };
            // SAFETY: inputs > 0, the active prefix is semi-normalized, and
            // DIF propagates implicit-zero supports until every requested
            // output is written. The root and private scratch match the ring.
            unsafe {
                SsaTransform::fft_recursive_dif_with_executor(
                    matrix, len, root, scratch, inputs, &ctx,
                );
            }
            return;
        }
        let half = len >> 1;
        let (low, high) = matrix.split_at_mut(matrix.len() >> 1);
        let child_inputs = inputs.min(half);
        let high_inputs = inputs.saturating_sub(half);
        // SAFETY: root*len=period and len>=2, so 2*root<=period.
        let child_root = unsafe { root.unchecked_mul(2) };
        let need_high = outputs > half;
        // SAFETY: complete disjoint child matrices carry exactly these active
        // prefixes; the stage never reads their implicit zero tails.
        unsafe {
            self.forward_columns(
                low,
                high,
                root,
                child_inputs,
                high_inputs,
                need_high,
                executor,
                scratch,
            );
        }
        if !need_high {
            // SAFETY: sums occupy the child's active prefix; omitted inputs are implicit zero.
            unsafe {
                self.forward(
                    low,
                    half,
                    child_root,
                    child_inputs,
                    outputs,
                    executor,
                    scratch,
                );
            }
        } else if SsaTransform::should_parallelize(
            half,
            self.cl.get(),
            self.cl.get(),
            scratch.len(),
            executor.parallelism().get(),
        ) {
            let (first, second) = scratch.split_at_mut(scratch.len() >> 1);
            let ((), ()) = executor.join(
                // SAFETY: low and first are complete disjoint child partitions.
                || unsafe {
                    self.forward(low, half, child_root, child_inputs, half, executor, first);
                },
                // SAFETY: high and second are complete disjoint child partitions.
                || unsafe {
                    self.forward(
                        high,
                        half,
                        child_root,
                        child_inputs,
                        outputs.unchecked_sub(half),
                        executor,
                        second,
                    );
                },
            );
        } else {
            // SAFETY: complete child prefixes reuse the arena synchronously.
            unsafe {
                self.forward(low, half, child_root, child_inputs, half, executor, scratch);
                self.forward(
                    high,
                    half,
                    child_root,
                    child_inputs,
                    outputs.unchecked_sub(half),
                    executor,
                    scratch,
                );
            }
        }
    }

    /// Forms the requested DIF children, specializing sums and zero-high columns.
    ///
    /// Butterfly columns and implicit-zero copies run as separate uniform
    /// passes, so no kernel branches on the input support. Every shift arrives
    /// reduced through the column scheduler's running progression.
    ///
    /// # Safety
    /// Both matrices contain the same power-of-two number of complete slots.
    /// Their active prefixes satisfy `high_count <= low_count`; scratch
    /// is disjoint and complete, and root times the child length equals bits.
    #[expect(
        clippy::too_many_arguments,
        reason = "column pass schedules matrix pairs across multi-slot strides, active counts, and executor"
    )]
    pub unsafe fn forward_columns<E: ParallelExecutor>(
        &self,
        low: &mut [Limb],
        high: &mut [Limb],
        root: usize,
        low_count: usize,
        high_count: usize,
        need_high: bool,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        if !need_high {
            // SAFETY: only the complete nonzero high prefix participates in sums;
            // the column scheduler supplies disjoint initialized coefficient slots.
            unsafe {
                self.columns(
                    [low, high],
                    0..high_count,
                    root,
                    executor,
                    scratch,
                    &|left, right, _, _| SsaRing::add_in_place(left, right, self.bits),
                );
            }
            return;
        }
        if high_count != 0 {
            // The first column's root exponent is zero. Form its sum and
            // difference once, leaving only positive twiddles in the pass.
            // SAFETY: high_count>0 and high_count<=low_count establish both
            // complete first inputs. Their cl-limb slots are disjoint, and
            // add_sub may overwrite the consumed high input with its difference.
            unsafe {
                let left = low.get_unchecked_mut(..self.cl.get());
                let right = high.get_unchecked_mut(..self.cl.get());
                let difference = from_mut::<[Limb]>(right);
                SsaRing::add_sub(
                    left,
                    difference,
                    difference.cast::<Limb>(),
                    self.bits,
                    self.kernel,
                );
            }
        }
        // SAFETY: the scheduler supplies complete disjoint slots and private
        // scratch; high is read only within high_count. The butterfly range
        // starts at index one, so its shifts are positive. Every shift stays
        // below bits because each range lies below the half length.
        unsafe {
            self.columns(
                [low, high],
                1..high_count,
                root,
                executor,
                scratch,
                &|left, right, shift, work| {
                    let difference = from_mut::<[Limb]>(right);
                    SsaRing::add_sub(
                        left,
                        difference,
                        difference.cast::<Limb>(),
                        self.bits,
                        self.kernel,
                    );
                    SsaRing::shift_in_place(right, shift, self.bits, work);
                },
            );
            self.columns(
                [low, high],
                high_count..low_count,
                root,
                executor,
                scratch,
                &|left, right, shift, _| {
                    SsaRing::shift_from(right, left, shift, self.bits);
                },
            );
        }
    }

    /// Applies a ring-column kernel to exactly the selected coefficient range.
    ///
    /// The scheduler carries reduced twiddle exponents directly. Every column
    /// index lies below the half transform length, so its shift stays below
    /// the inner bit width with no per-column multiplication or modular
    /// reduction: one product establishes the range start and each step adds
    /// the root.
    ///
    /// # Safety
    /// The two matrices are disjoint equal complete coefficient spans. range
    /// lies within both and below the half transform length, so every running
    /// shift stays below the inner bit width. Scratch is disjoint and contains
    /// a full coefficient. The kernel accepts complete slots, their reduced
    /// shift, and private scratch, reading only inputs established by the
    /// caller's support proof.
    pub unsafe fn columns<E, F>(
        &self,
        matrices: [&mut [Limb]; 2],
        range: Range<usize>,
        root: usize,
        executor: &E,
        scratch: &mut [Limb],
        kernel: &F,
    ) where
        E: ParallelExecutor,
        F: Fn(&mut [Limb], &mut [Limb], usize, &mut [Limb]) + Sync,
    {
        if range.start >= range.end {
            return;
        }
        // SAFETY: the empty range returned, leaving count>0. The ordered
        // coefficient range lies within a matrix whose
        // complete limb span is representable, so both products and the
        // nonnegative coefficient difference are present on every pointer width.
        let (start, span, count) = unsafe {
            let count = range.end.unchecked_sub(range.start);
            (
                range.start.unchecked_mul(self.cl.get()),
                count.unchecked_mul(self.cl.get()),
                NonZeroUsize::new_unchecked(count),
            )
        };
        let [low, high] = matrices;
        // SAFETY: range.end*cl lies in both complete matrix spans. The two
        // selected prefixes stay disjoint and cover exactly range.len() slots.
        let columns = unsafe {
            let (_, low_tail) = low.split_at_mut_unchecked(start);
            let (_, high_tail) = high.split_at_mut_unchecked(start);
            [
                low_tail.split_at_mut_unchecked(span).0,
                high_tail.split_at_mut_unchecked(span).0,
            ]
        };
        // One product establishes the running shift; every column stays below
        // the half length, so the progression never wraps past the bit width.
        // SAFETY: range.start lies below the half transform length.
        let start_twiddle = unsafe { range.start.unchecked_mul(root) };
        // SAFETY: the selected spans contain count complete independent columns;
        // all forks preserve these boundaries and own private staging slots.
        unsafe {
            self.column_range(
                columns,
                count,
                start_twiddle,
                root,
                executor,
                scratch,
                kernel,
            );
        }
    }

    /// Forks a column pass using local work and private scratch as its budget.
    ///
    /// # Safety
    /// Equal matrix spans contain `count` complete columns whose running shifts start
    /// at `twiddle` and advance by `step` without wrapping past the inner bit
    /// width. Scratch contains a coefficient, and the callback satisfies the
    /// column kernel contract above.
    #[expect(
        clippy::too_many_arguments,
        reason = "Carries the admitted column count and reduced twiddle progression through private matrix and worker partitions"
    )]
    unsafe fn column_range<E, F>(
        &self,
        matrices: [&mut [Limb]; 2],
        count: NonZeroUsize,
        twiddle: usize,
        step: usize,
        executor: &E,
        scratch: &mut [Limb],
        kernel: &F,
    ) where
        E: ParallelExecutor,
        F: Fn(&mut [Limb], &mut [Limb], usize, &mut [Limb]) + Sync,
    {
        let [low, high] = matrices;
        let cl = self.cl.get();
        if SsaTransform::should_parallelize(
            count.get(),
            cl,
            cl,
            scratch.len(),
            executor.parallelism().get(),
        ) {
            let half = count.get() >> 1;
            // SAFETY: should_parallelize requires count>=2; thus 0<half<count.
            // Both child counts are positive and cover the admitted parent.
            let (left_count, right_count) = unsafe {
                (
                    NonZeroUsize::new_unchecked(half),
                    NonZeroUsize::new_unchecked(count.get().unchecked_sub(half)),
                )
            };
            // SAFETY: half<=count and count*cl equals the admitted matrix span.
            let offset = unsafe { half.unchecked_mul(cl) };
            // SAFETY: both equal spans contain count complete columns; half
            // selects an interior coefficient boundary in each exclusive span.
            let ((low_left, low_right), (high_left, high_right)) = unsafe {
                (
                    low.split_at_mut_unchecked(offset),
                    high.split_at_mut_unchecked(offset),
                )
            };
            // SAFETY: halving selects an in-bounds scratch boundary.
            // should_parallelize admits at least two complete cl-limb arenas,
            // so both child partitions retain a full coefficient.
            let (first, second) = unsafe { scratch.split_at_mut_unchecked(scratch.len() >> 1) };
            // The right progression continues without reduction: both halves
            // stay below the half transform length by the caller's bound.
            // SAFETY: half*step stays within the validated twiddle progression.
            let next_twiddle = unsafe { twiddle.unchecked_add(half.unchecked_mul(step)) };
            let ((), ()) = executor.join(
                // SAFETY: first column halves own one complete private staging arena.
                || unsafe {
                    self.column_range(
                        [low_left, high_left],
                        left_count,
                        twiddle,
                        step,
                        executor,
                        first,
                        kernel,
                    );
                },
                // SAFETY: remaining column halves own the disjoint second arena.
                || unsafe {
                    self.column_range(
                        [low_right, high_right],
                        right_count,
                        next_twiddle,
                        step,
                        executor,
                        second,
                        kernel,
                    );
                },
            );
            return;
        }
        let mut shift = twiddle;
        for index in 0..count.get() {
            // SAFETY: the admitted equal matrices each hold count complete
            // cl-limb slots. The indexed coefficient windows are disjoint;
            // the callback's support proof establishes every input it reads.
            let (left, right) = unsafe {
                let offset = index.unchecked_mul(cl);
                let end = offset.unchecked_add(cl);
                (
                    low.get_unchecked_mut(offset..end),
                    high.get_unchecked_mut(offset..end),
                )
            };
            // The running shift stays below the inner bit width by the
            // caller's half-length bound; its only use is a reduced exponent.
            kernel(left, right, shift, scratch);
            // SAFETY: the final increment is at most the half-length root
            // product, bits; earlier increments are strictly smaller.
            shift = unsafe { shift.unchecked_add(step) };
        }
    }
}
