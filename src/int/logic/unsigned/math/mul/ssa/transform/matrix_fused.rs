//! Row fusion of forward TFT, pointwise multiplication, and inverse TFT.
//!
//! Complete rows are independent. Each row finishes its transform, product,
//! and inverse before the column pass, keeping its coefficient working set local.

#![expect(
    unsafe_code,
    reason = "The accepted plan bounds complete matrix partitions, row supports, and private execution arenas"
)]

use core::num::NonZeroUsize;

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{
    CoefficientView, Limb, LimbOutput, MulTransformPlan, SsaCoefficients, SsaPlan, SsaPointwise,
    SsaTransform, TruncatedTransform,
};

/// Geometry and workspace bounds shared by every complete-row batch.
struct FusedRows<'plan> {
    width: NonZeroUsize,
    row_root: usize,
    supports: [usize; 2],
    row_limbs: usize,
    row_work: usize,
    minimum_scratch: usize,
    plan: &'plan MulTransformPlan,
    transform: &'plan TruncatedTransform,
}

impl SsaTransform {
    /// Executes row-wise fused matrix truncated multiplication.
    ///
    /// # Safety
    /// Inputs have nonempty active chunk prefixes within `plan.transform_len`.
    /// `plan.transform_len >= 4` and `plan.transform_len/2 < support < plan.transform_len`.
    /// The product polynomial has degree strictly below `support`.
    /// `transform` matches the ring modulus, limb width, and root geometry of `plan`.
    /// `scratch` satisfies `plan.transform_mul_scratch(parallelism)`.
    /// All buffers are disjoint. The destination holds a complete guarded ring
    /// coefficient or the proven exact product, as in the enclosing transform entry.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Fused matrix multiplication binds operand staging, fused row kernels, and reconstruction"
    )]
    pub unsafe fn fused_matrix_truncated_mul<E: ParallelExecutor>(
        left: &[Limb],
        right: &[Limb],
        dst: &mut [impl LimbOutput],
        active_left_chunks: usize,
        active_right_chunks: usize,
        support: usize,
        plan: &MulTransformPlan,
        transform: &TruncatedTransform,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        // SAFETY: scratch has at least plan.mat_limbs limbs for left_matrix.
        let (left_matrix, after_left) =
            unsafe { scratch.split_at_mut_unchecked(plan.mat_limbs.get()) };

        let cl = plan.inner_cl.get();
        {
            // SAFETY: after_left has at least plan.mat_limbs limbs for right_matrix and work.
            let (right_matrix, work) =
                unsafe { after_left.split_at_mut_unchecked(plan.mat_limbs.get()) };

            // SAFETY: active chunk counts <= plan.transform_len, and mat_limbs covers all chunks.
            let (active_left_matrix, active_right_matrix) = unsafe {
                let active_left_len = active_left_chunks.unchecked_mul(cl);
                let active_right_len = active_right_chunks.unchecked_mul(cl);
                (
                    left_matrix.get_unchecked_mut(..active_left_len),
                    right_matrix.get_unchecked_mut(..active_right_len),
                )
            };

            // SAFETY: K>=4 and the accepted matrix span K*cl fits usize.
            let minimum_fork = unsafe { cl.unchecked_mul(4) };
            if executor.parallelism().get() > 1 && work.len() >= minimum_fork {
                let (first_work, second_work) = work.split_at_mut(work.len() >> 1);
                let ((), ()) = executor.join(
                    // SAFETY: left matrix staging owns active_left_matrix and first_work disjointly.
                    || unsafe {
                        SsaCoefficients::split_twisted_with_executor(
                            left,
                            active_left_matrix,
                            active_left_chunks,
                            plan.chunk_bits,
                            plan.inner_cl,
                            plan.periods,
                            plan.twist_step_half,
                            executor,
                            first_work,
                        );
                    },
                    // SAFETY: right matrix staging owns active_right_matrix and second_work disjointly.
                    || unsafe {
                        SsaCoefficients::split_twisted_with_executor(
                            right,
                            active_right_matrix,
                            active_right_chunks,
                            plan.chunk_bits,
                            plan.inner_cl,
                            plan.periods,
                            plan.twist_step_half,
                            executor,
                            second_work,
                        );
                    },
                );
            } else {
                // SAFETY: sequential staging initializes each matrix prefix disjointly.
                unsafe {
                    SsaCoefficients::split_twisted_with_executor(
                        left,
                        active_left_matrix,
                        active_left_chunks,
                        plan.chunk_bits,
                        plan.inner_cl,
                        plan.periods,
                        plan.twist_step_half,
                        executor,
                        work,
                    );
                    SsaCoefficients::split_twisted_with_executor(
                        right,
                        active_right_matrix,
                        active_right_chunks,
                        plan.chunk_bits,
                        plan.inner_cl,
                        plan.periods,
                        plan.twist_step_half,
                        executor,
                        work,
                    );
                }
            }

            // SAFETY: left_matrix and right_matrix each contain plan.mat_limbs complete slots.
            let (width, row_root, col_support_a, col_support_b, n1, n2) = unsafe {
                let mut view_a =
                    CoefficientView::new(left_matrix, plan.transform_len, plan.inner_cl);
                let (width, row_root, _, col_support_a, n1, n2) = transform.matrix_forward_columns(
                    &mut view_a,
                    plan.twist_step_half,
                    0,
                    active_left_chunks,
                    support,
                    executor,
                    work,
                );
                let mut view_b =
                    CoefficientView::new(right_matrix, plan.transform_len, plan.inner_cl);
                let (_, _, _, col_support_b, _, _) = transform.matrix_forward_columns(
                    &mut view_b,
                    plan.twist_step_half,
                    0,
                    active_right_chunks,
                    support,
                    executor,
                    work,
                );
                (width, row_root, col_support_a, col_support_b, n1, n2)
            };

            // SAFETY: K>=4 gives width<=K/2<support, so n1=floor(support/width)>0.
            // Also n1*width<=support<K and K*cl fits. The column pass initializes
            // every coefficient in the complete row prefixes.
            unsafe {
                let row_limbs = width.get().unchecked_mul(cl);
                let complete_limbs = n1.unchecked_mul(row_limbs);
                let left_complete = left_matrix.get_unchecked_mut(..complete_limbs);
                let right_complete = right_matrix.get_unchecked_mut(..complete_limbs);
                let rows = FusedRows {
                    width,
                    row_root,
                    supports: [col_support_a, col_support_b],
                    row_limbs,
                    row_work: width
                        .get()
                        .saturating_mul(SsaPlan::basecase_product_cost(cl)),
                    minimum_scratch: plan.pointwise().scratch_len.get().max(cl),
                    plan,
                    transform,
                };
                rows.run(
                    NonZeroUsize::new_unchecked(n1),
                    left_complete,
                    right_complete,
                    executor,
                    work,
                );
            }

            if n2 != 0 {
                // SAFETY: row n1 exists since n2 != 0 implies n1 < rows; bounds are within mat_limbs.
                unsafe {
                    let row_limbs = width.get().unchecked_mul(cl);
                    let partial_start = n1.unchecked_mul(row_limbs);
                    let partial_end = partial_start.unchecked_add(row_limbs);
                    let partial_a = left_matrix.get_unchecked_mut(partial_start..partial_end);
                    let partial_b = right_matrix.get_unchecked_mut(partial_start..partial_end);
                    let partial_left_view =
                        CoefficientView::new(partial_a, width.get(), plan.inner_cl);
                    let partial_right_view =
                        CoefficientView::new(partial_b, width.get(), plan.inner_cl);
                    transform.matrix_forward(
                        partial_left_view,
                        row_root,
                        0,
                        col_support_a,
                        n2,
                        executor,
                        work,
                    );
                    transform.matrix_forward(
                        partial_right_view,
                        row_root,
                        0,
                        col_support_b,
                        n2,
                        executor,
                        work,
                    );
                    let prefix_limbs = n2.unchecked_mul(cl);
                    let leaf_budget = plan.pointwise_leaf_count(executor.parallelism().get());
                    SsaPointwise::pointwise_multiply_with_executor(
                        partial_a.get_unchecked_mut(..prefix_limbs),
                        partial_b.get_unchecked_mut(..prefix_limbs),
                        n2,
                        leaf_budget,
                        plan.pointwise(),
                        executor,
                        work,
                    );
                }
            }

            // SAFETY: complete rows are inverted, and partial row has n2 frequencies established.
            unsafe {
                let mut inv_view =
                    CoefficientView::new(left_matrix, plan.transform_len, plan.inner_cl);
                transform.matrix_inverse_columns_and_partial(
                    &mut inv_view,
                    plan.twist_step_half,
                    0,
                    support,
                    support,
                    false,
                    executor,
                    work,
                );
            }
        }

        // SAFETY: parallel_slots<=K, so slots*cl<=mat_limbs. The accepted
        // reconstruction phase reserves that prefix in the released right arena.
        let (twiddle, recon_scratch) = unsafe {
            let twiddle_len = plan
                .inner_cl
                .get()
                .unchecked_mul(plan.parallel_slots(executor.parallelism().get()));
            after_left.split_at_mut_unchecked(twiddle_len)
        };

        let inverse_twist = plan.inverse_twist();
        // SAFETY: prefix has support complete coefficients, bounded by plan.mat_limbs.
        let prefix = unsafe {
            let prefix_len = support.unchecked_mul(cl);
            left_matrix.get_unchecked_mut(..prefix_len)
        };
        // SAFETY: the inverse establishes support complete K-scaled coefficients.
        // The plan bounds their magnitudes and reserves the inverse-twist and
        // accumulator arenas; dst satisfies the enclosing exact/guarded contract.
        unsafe {
            SsaCoefficients::reconstruct(
                prefix,
                plan.transform_len,
                plan.chunk_bits,
                plan.inner_bits,
                plan.modulus_bits,
                dst,
                recon_scratch,
                Some((inverse_twist, twiddle)),
                executor,
            );
        }
    }
}

impl FusedRows<'_> {
    /// Partitions independent complete rows and their private execution arenas.
    ///
    /// # Safety
    /// Both disjoint matrices contain exactly `row_count` complete initialized rows
    /// with the declared supports. Scratch contains at least `minimum_scratch` limbs.
    unsafe fn run<E: ParallelExecutor>(
        &self,
        row_count: NonZeroUsize,
        left_rows: &mut [Limb],
        right_rows: &mut [Limb],
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let count = row_count.get();
        let has_scratch = (scratch.len() >> 1) >= self.minimum_scratch;
        if count > 1
            && SsaTransform::has_parallel_work(count, self.row_work, executor.parallelism().get())
            && has_scratch
        {
            let half = count >> 1;
            // SAFETY: count>1 gives 0<half<count, so both child counts are positive.
            // Both spans contain count complete rows; their represented lengths
            // bound the split and the remaining count.
            let (left_low, left_high, right_low, right_high, left_count, right_count) = unsafe {
                let remaining = count.unchecked_sub(half);
                let split_limbs = half.unchecked_mul(self.row_limbs);
                let (l_low, l_high) = left_rows.split_at_mut_unchecked(split_limbs);
                let (r_low, r_high) = right_rows.split_at_mut_unchecked(split_limbs);
                (
                    l_low,
                    l_high,
                    r_low,
                    r_high,
                    NonZeroUsize::new_unchecked(half),
                    NonZeroUsize::new_unchecked(remaining),
                )
            };
            let (scratch_low, scratch_high) = scratch.split_at_mut(scratch.len() >> 1);
            let ((), ()) = executor.join(
                // SAFETY: lower half operates on disjoint low row spans and private scratch.
                || unsafe {
                    self.run(left_count, left_low, right_low, executor, scratch_low);
                },
                // SAFETY: upper half operates on disjoint high row spans and private scratch.
                || unsafe {
                    self.run(right_count, left_high, right_high, executor, scratch_high);
                },
            );
            return;
        }
        let cl = self.plan.inner_cl;
        let pointwise = self.plan.pointwise();
        for index in 0..count {
            // SAFETY: index<count and both slices contain count*row_limbs
            // initialized limbs. The start and end lie within those spans;
            // each row borrow ends before the next iteration.
            let (row_a, row_b) = unsafe {
                let start = index.unchecked_mul(self.row_limbs);
                let end = start.unchecked_add(self.row_limbs);
                (
                    left_rows.get_unchecked_mut(start..end),
                    right_rows.get_unchecked_mut(start..end),
                )
            };
            // SAFETY: row_a and row_b are contiguous row slices of width complete slots.
            let (left_view, right_view) = unsafe {
                (
                    CoefficientView::new(row_a, self.width.get(), cl),
                    CoefficientView::new(row_b, self.width.get(), cl),
                )
            };
            // SAFETY: row transforms and pointwise multiply execute sequentially with private scratch.
            unsafe {
                self.transform.matrix_forward(
                    left_view,
                    self.row_root,
                    0,
                    self.supports[0],
                    self.width.get(),
                    &SequentialExecutor,
                    scratch,
                );
                self.transform.matrix_forward(
                    right_view,
                    self.row_root,
                    0,
                    self.supports[1],
                    self.width.get(),
                    &SequentialExecutor,
                    scratch,
                );
                SsaPointwise::pointwise_multiply_with_executor(
                    row_a,
                    row_b,
                    self.width.get(),
                    NonZeroUsize::MIN,
                    pointwise,
                    &SequentialExecutor,
                    scratch,
                );
                let row_a_inv_view = CoefficientView::new(row_a, self.width.get(), cl);
                self.transform.matrix_inverse(
                    row_a_inv_view,
                    self.row_root,
                    0,
                    self.width.get(),
                    self.width.get(),
                    false,
                    &SequentialExecutor,
                    scratch,
                );
            }
        }
    }
}
