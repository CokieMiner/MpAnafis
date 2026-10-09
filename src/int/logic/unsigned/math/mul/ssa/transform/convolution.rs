//! Matched DIF/product/DIT subtrees with one prepared coefficient strategy.
//!
//! The traversal completes one cache-resident subtree at a time: the forward
//! stages, pointwise products, and inverse stages of a subtree finish before
//! its neighbor begins, so the matrix is never swept between phases.

#![expect(
    unsafe_code,
    reason = "Transform geometry and exact per-worker arenas bound recursive disjoint subtree partitions"
)]

use core::{array::from_fn, mem::size_of, num::NonZeroUsize, ops::Div};

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{ArchKernels, CACHE_BLOCK_BYTES, Limb, SsaTransform, TransformContext};

/// Dimensions and architecture selection shared by every matching subtree.
pub struct Convolution {
    ctx: TransformContext<'static, SequentialExecutor>,
    leaf_len: usize,
    arena_unit: NonZeroUsize,
}

impl Convolution {
    /// Sizes a leaf from the simultaneously live matrices and the exact arena.
    ///
    /// # Safety
    /// The admitted ring is positive and has a representable half-bit period;
    /// `arena_len>=work_len`. INPUTS is the positive fixed input arity 1, 2, or 3.
    pub unsafe fn new<const INPUTS: usize>(
        bits: usize,
        work_len: NonZeroUsize,
        requested: NonZeroUsize,
        arena_len: usize,
        transform_len: usize,
    ) -> (Self, NonZeroUsize) {
        let available = arena_len.div(work_len).min(requested.get());
        // SAFETY: arena_len>=work_len>0 gives at least one complete arena;
        // the nonzero executor request preserves this positive worker budget.
        let workers = unsafe { NonZeroUsize::new_unchecked(available) };
        let cache_limbs = CACHE_BLOCK_BYTES.div_euclid(size_of::<Limb>());
        let ctx = TransformContext::new(
            bits,
            ArchKernels::selected_add_sub_from_limbs_unchecked(),
            &SequentialExecutor,
        );
        let cache_leaf = cache_limbs
            .saturating_sub(work_len.get())
            .div(ctx.cl)
            .div_euclid(INPUTS)
            .max(4);
        let leaf_len = if workers.get() > 1 {
            cache_leaf.min(transform_len.div(workers).max(4))
        } else {
            cache_leaf
        };
        (
            Self {
                ctx,
                leaf_len,
                arena_unit: work_len,
            },
            workers,
        )
    }

    /// Traverses matching radix-four subtrees, retaining the input/output arity
    /// as compile-time constants. The first `OUTPUTS` matrices hold the results.
    ///
    /// # Safety
    /// Each matrix contains `len` initialized coefficients for this ring and
    /// root, and `scratch` holds one complete arena per budgeted worker.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "The fused traversal carries its range, root, budget, executor, arena, and callback through direct forward/child/inverse stages without a forwarding stage helper"
    )]
    pub unsafe fn run<const INPUTS: usize, const OUTPUTS: usize, E: ParallelExecutor, P>(
        &self,
        mut matrices: [&mut [Limb]; INPUTS],
        len: usize,
        root: usize,
        budget: NonZeroUsize,
        executor: &E,
        scratch: &mut [Limb],
        pointwise: &P,
    ) where
        P: Fn([&mut [Limb]; INPUTS], usize, &mut [Limb]) + Sync,
    {
        if len <= self.leaf_len {
            // SAFETY: all matrix spans are complete and the retained callback
            // and workspace match their coefficient ring.
            unsafe {
                self.leaf::<INPUTS, OUTPUTS, P>(matrices, len, root, scratch, pointwise);
            }
            return;
        }
        let quarter = len >> 2;
        // SAFETY: the resident leaf returned for len<=leaf_len>=4. The admitted
        // power-of-two parent has len>=8 and four nonempty coefficient quarters.
        let quarter_width = unsafe { NonZeroUsize::new_unchecked(quarter) };
        // SAFETY: len>leaf_len>=4 and every matrix has exactly len*cl
        // limbs, with cl positive. Each of its four complete quarters is nonempty.
        let span = unsafe { NonZeroUsize::new_unchecked(matrices[0].len() >> 2) };
        let imaginary_shift = self.ctx.mod_bits >> 1;
        for matrix in &mut matrices {
            // SAFETY: the complete parent has four disjoint initialized
            // quarter_width*cl spans and a primitive root. Its entire support
            // is requested, so the dense leaf needs no sparse-stage dispatch.
            // self.ctx is sequential: the outer convolution owns all forks.
            unsafe {
                let (first, after_first) = matrix.split_at_mut_unchecked(span.get());
                let (second, after_second) = after_first.split_at_mut_unchecked(span.get());
                let (third, fourth) = after_second.split_at_mut_unchecked(span.get());
                SsaTransform::dif_radix4_dense_block(
                    [first, second, third, fourth],
                    quarter_width,
                    0,
                    root,
                    imaginary_shift,
                    scratch,
                    &self.ctx,
                );
            }
        }
        // SAFETY: len is a power of two above leaf_len>=4, so len>=8.
        // The principal exponent satisfies root*len=period, hence 4*root<period.
        let next_root = unsafe { root.unchecked_mul(4) };
        // SAFETY: budget*arena_unit <= scratch.len() by the worker reservation.
        let half_arena = unsafe { (budget.get() >> 1).unchecked_mul(self.arena_unit.get()) };
        if budget.get() >= 2
            && SsaTransform::should_parallelize(
                len >> 1,
                self.ctx.cl.get(),
                self.ctx.cl.get(),
                half_arena,
                2,
            )
        {
            let mut quarters = matrices
                .each_mut()
                .map(|matrix| matrix.chunks_exact_mut(span.get()));
            let [q0, q1, q2, q3]: [[&mut [Limb]; INPUTS]; 4] = from_fn(|_| {
                let mut inputs = quarters.iter_mut();
                from_fn(|_| {
                    // SAFETY: each matrix has exactly four nonempty spans;
                    // the outer array consumes precisely four from each
                    // iterator. The inner array consumes exactly INPUTS
                    // iterators from the equally sized quarters array.
                    unsafe { inputs.next().unwrap_unchecked().next().unwrap_unchecked() }
                })
            });
            // SAFETY: budget>=2 gives a positive first half and remainder.
            let first_budget = unsafe { NonZeroUsize::new_unchecked(budget.get() >> 1) };
            // SAFETY: first_budget<=budget and each worker owns arena_unit
            // limbs. The parent arena contains both exact disjoint shares.
            let (second_budget, (first_arena, second_arena)) = unsafe {
                let first_len = first_budget.get().unchecked_mul(self.arena_unit.get());
                (
                    NonZeroUsize::new_unchecked(budget.get().unchecked_sub(first_budget.get())),
                    scratch.split_at_mut_unchecked(first_len),
                )
            };
            let ((), ()) = executor.join(
                // SAFETY: first quarters, budget, and arena are disjoint from
                // the second arm; parent stages completed before the fork.
                || unsafe {
                    self.run_pair::<INPUTS, OUTPUTS, E, P>(
                        q0,
                        q1,
                        quarter,
                        next_root,
                        first_budget,
                        executor,
                        first_arena,
                        pointwise,
                    );
                },
                // SAFETY: second quarters, budget, and arena are disjoint from
                // the first arm.
                || unsafe {
                    self.run_pair::<INPUTS, OUTPUTS, E, P>(
                        q2,
                        q3,
                        quarter,
                        next_root,
                        second_budget,
                        executor,
                        second_arena,
                        pointwise,
                    );
                },
            );
        } else {
            for index in 0_usize..4 {
                // SAFETY: index<4 and four spans partition every complete matrix.
                let (start, end) = unsafe {
                    let start = index.unchecked_mul(span.get());
                    (start, start.unchecked_add(span.get()))
                };
                let children = matrices.each_mut().map(|matrix| {
                    // SAFETY: start..end is one complete quarter of this matrix.
                    unsafe { matrix.get_unchecked_mut(start..end) }
                });
                // SAFETY: the fourth-powered root is primitive for each child;
                // synchronous recursion releases the arena before its next use.
                unsafe {
                    self.run::<INPUTS, OUTPUTS, E, P>(
                        children,
                        quarter,
                        next_root,
                        NonZeroUsize::MIN,
                        executor,
                        scratch,
                        pointwise,
                    );
                }
            }
        }
        // SAFETY: root is positive and below period/4; imaginary_shift is n/2.
        // Their inverse exponents are positive exact differences below period.
        let (inverse_root, inverse_imaginary) = unsafe {
            (
                self.ctx.period.get().unchecked_sub(root),
                self.ctx.period.get().unchecked_sub(imaginary_shift),
            )
        };
        for matrix in matrices.iter_mut().take(OUTPUTS) {
            // SAFETY: completed children provide full frequency support in
            // four disjoint initialized quarter_width*cl spans. Staging is
            // private and the retained context is sequential; all outer forks
            // completed before this parent inverse consumes their products.
            unsafe {
                let (first, after_first) = matrix.split_at_mut_unchecked(span.get());
                let (second, after_second) = after_first.split_at_mut_unchecked(span.get());
                let (third, fourth) = after_second.split_at_mut_unchecked(span.get());
                SsaTransform::dit_radix4_dense_block(
                    [first, second, third, fourth],
                    quarter_width,
                    0,
                    inverse_root,
                    inverse_imaginary,
                    scratch,
                    &self.ctx,
                );
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "A quarter-pair carries both ranges, root, budget, executor, arena, and callback"
    )]
    unsafe fn run_pair<const INPUTS: usize, const OUTPUTS: usize, E: ParallelExecutor, P>(
        &self,
        first: [&mut [Limb]; INPUTS],
        second: [&mut [Limb]; INPUTS],
        quarter: usize,
        root: usize,
        budget: NonZeroUsize,
        executor: &E,
        scratch: &mut [Limb],
        pointwise: &P,
    ) where
        P: Fn([&mut [Limb]; INPUTS], usize, &mut [Limb]) + Sync,
    {
        // SAFETY: budget*arena_unit is within the caller's reserved arena.
        let half_arena = unsafe { (budget.get() >> 1).unchecked_mul(self.arena_unit.get()) };
        if budget.get() >= 2
            && SsaTransform::should_parallelize(
                quarter,
                self.ctx.cl.get(),
                self.ctx.cl.get(),
                half_arena,
                2,
            )
        {
            // SAFETY: budget>=2 gives a positive first half and remainder.
            let first_budget = unsafe { NonZeroUsize::new_unchecked(budget.get() >> 1) };
            // SAFETY: the parent budget splits into two nonzero shares, and
            // their exact workspaces partition the reserved initialized arena.
            let (second_budget, (first_arena, second_arena)) = unsafe {
                let first_len = first_budget.get().unchecked_mul(self.arena_unit.get());
                (
                    NonZeroUsize::new_unchecked(budget.get().unchecked_sub(first_budget.get())),
                    scratch.split_at_mut_unchecked(first_len),
                )
            };
            let ((), ()) = executor.join(
                // SAFETY: first quarters, budget, and arena are disjoint from
                // the second child.
                || unsafe {
                    self.run::<INPUTS, OUTPUTS, E, P>(
                        first,
                        quarter,
                        root,
                        first_budget,
                        executor,
                        first_arena,
                        pointwise,
                    );
                },
                // SAFETY: second quarters, budget, and arena are disjoint.
                || unsafe {
                    self.run::<INPUTS, OUTPUTS, E, P>(
                        second,
                        quarter,
                        root,
                        second_budget,
                        executor,
                        second_arena,
                        pointwise,
                    );
                },
            );
            return;
        }
        // SAFETY: sequential children reuse the complete arena one after the
        // other; every borrow ends before the next child begins.
        unsafe {
            self.run::<INPUTS, OUTPUTS, E, P>(
                first,
                quarter,
                root,
                NonZeroUsize::MIN,
                executor,
                scratch,
                pointwise,
            );
            self.run::<INPUTS, OUTPUTS, E, P>(
                second,
                quarter,
                root,
                NonZeroUsize::MIN,
                executor,
                scratch,
                pointwise,
            );
        }
    }

    /// Executes the remaining complete transforms inside one resident leaf.
    ///
    /// # Safety
    /// Inherits the complete matrices, root, callback, and arena contracts of run.
    unsafe fn leaf<const INPUTS: usize, const OUTPUTS: usize, P>(
        &self,
        mut matrices: [&mut [Limb]; INPUTS],
        len: usize,
        root: usize,
        scratch: &mut [Limb],
        pointwise: &P,
    ) where
        P: Fn([&mut [Limb]; INPUTS], usize, &mut [Limb]),
    {
        for matrix in &mut matrices {
            // SAFETY: each complete initialized matrix shares the primitive
            // root; the selected architecture kernel is valid on this target.
            unsafe {
                SsaTransform::fft_recursive_dif_with_executor(
                    matrix, len, root, scratch, len, &self.ctx,
                );
            }
        }
        pointwise(
            matrices.each_mut().map(|matrix| &mut **matrix),
            len,
            scratch,
        );
        for matrix in matrices.iter_mut().take(OUTPUTS) {
            // SAFETY: the pointwise callback establishes every frequency value
            // required by this complete inverse. Each output is inverted once.
            unsafe {
                SsaTransform::fft_recursive_dit_with_executor(
                    matrix, len, root, scratch, len, &self.ctx,
                );
            }
        }
    }
}
