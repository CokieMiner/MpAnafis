//! Prepared product, square, and paired-product convolution entry points.

#![expect(
    unsafe_code,
    reason = "Admitted coefficient plans establish complete disjoint matrices and per-worker arenas"
)]

use core::num::NonZeroUsize;

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::{Convolution, Limb, PointwiseMulPlan, PointwiseSquarePlan, SsaPointwise, SsaTransform};

impl SsaTransform {
    /// Completes matched forward, product, and inverse subtrees.
    ///
    /// # Safety
    /// Both matrices contain `transform_len` complete semi-normalized
    /// coefficients in the plan's ring. The length is a power of two at least
    /// four, `root_shift` defines its root, and `arena` holds at least
    /// `plan.scratch_len` initialized limbs. All buffers are disjoint.
    pub unsafe fn convolve_subtrees<E: ParallelExecutor>(
        left: &mut [Limb],
        right: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        plan: &PointwiseMulPlan,
        executor: &E,
        arena: &mut [Limb],
    ) {
        // SAFETY: the admitted ring and complete arena satisfy the workspace
        // constructor; its fixed arity accounts for both live matrices.
        let (convolution, active_workers) = unsafe {
            Convolution::new::<2>(
                plan.bits,
                plan.scratch_len,
                executor.parallelism(),
                arena.len(),
                transform_len,
            )
        };
        // SAFETY: the matrices contain the admitted transform and the prepared
        // strategy matches every leaf's ring. Partitioning retains one arena
        // per worker; each leaf executes with one worker in its private arena.
        unsafe {
            convolution.run::<2, 1, E, _>(
                [left, right],
                transform_len,
                root_shift,
                active_workers,
                executor,
                arena,
                &|[leaf_left, leaf_right], len, leaf_work| {
                    SsaPointwise::pointwise_multiply_with_executor(
                        leaf_left,
                        leaf_right,
                        len,
                        NonZeroUsize::MIN,
                        plan,
                        &SequentialExecutor,
                        leaf_work,
                    );
                },
            );
        }
    }

    /// Completes matched forward, square, and inverse subtrees.
    ///
    /// # Safety
    /// The matrix contains a complete semi-normalized power-of-two transform
    /// of length at least four in the plan's ring for `root_shift`. The disjoint
    /// initialized `arena` holds at least `plan.scratch_len` limbs.
    pub unsafe fn convolve_square_subtrees<E: ParallelExecutor>(
        matrix: &mut [Limb],
        transform_len: usize,
        root_shift: usize,
        plan: &PointwiseSquarePlan,
        executor: &E,
        arena: &mut [Limb],
    ) {
        // SAFETY: the admitted ring and complete square arena satisfy the
        // workspace constructor with one live input matrix.
        let (convolution, active_workers) = unsafe {
            Convolution::new::<1>(
                plan.bits,
                plan.scratch_len,
                executor.parallelism(),
                arena.len(),
                transform_len,
            )
        };
        // SAFETY: the complete matrix, retained square strategy, and private
        // per-worker partitions describe the same coefficient ring.
        unsafe {
            convolution.run::<1, 1, E, _>(
                [matrix],
                transform_len,
                root_shift,
                active_workers,
                executor,
                arena,
                &|[leaf_matrix], len, leaf_work| {
                    SsaPointwise::pointwise_square_with_executor(
                        leaf_matrix,
                        len,
                        NonZeroUsize::MIN,
                        plan,
                        &SequentialExecutor,
                        leaf_work,
                    );
                },
            );
        }
    }

    /// Completes three forwards, two shared products, and two inverses.
    ///
    /// # Safety
    /// The disjoint matrices contain complete semi-normalized power-of-two
    /// transforms of length at least four in the plan's ring for `root_shift`.
    /// The disjoint initialized `arena` holds at least `plan.scratch_len` limbs.
    pub unsafe fn convolve_pair_subtrees<E: ParallelExecutor>(
        matrices: [&mut [Limb]; 3],
        transform_len: usize,
        root_shift: usize,
        plan: &PointwiseMulPlan,
        executor: &E,
        arena: &mut [Limb],
    ) {
        // SAFETY: the admitted ring and complete product arena satisfy the
        // workspace constructor with three live input matrices.
        let (convolution, active_workers) = unsafe {
            Convolution::new::<3>(
                plan.bits,
                plan.scratch_len,
                executor.parallelism(),
                arena.len(),
                transform_len,
            )
        };
        // SAFETY: the matrices and shared product strategy have matching ring
        // dimensions. Leaf partitions are disjoint and retain complete arenas.
        unsafe {
            convolution.run::<3, 2, E, _>(
                matrices,
                transform_len,
                root_shift,
                active_workers,
                executor,
                arena,
                &|[leaf_a, leaf_b, leaf_x], len, leaf_work| {
                    SsaPointwise::pointwise_multiply_pair_with_executor(
                        [leaf_a, leaf_b, leaf_x],
                        len,
                        NonZeroUsize::MIN,
                        plan,
                        &SequentialExecutor,
                        leaf_work,
                    );
                },
            );
        }
    }
}
