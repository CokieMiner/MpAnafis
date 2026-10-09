//! Paired coefficient products with one normalization of the shared operand.

#![expect(
    unsafe_code,
    reason = "Admitted complete coefficient matrices and private leaf arenas bound paired traversal and preserve the shared operand"
)]

use core::num::NonZeroUsize;

use crate::parallel::ParallelExecutor;

use super::{
    Limb, PointwiseMulPlan, PointwiseMulStrategy, SsaPlan, SsaPointwise, SsaRing, SsaTransform,
};

/// Fixed coefficient and arena widths carried through paired range execution.
struct PairWork {
    bits: usize,
    cl: NonZeroUsize,
    leaf_scratch: NonZeroUsize,
}

impl SsaPointwise {
    /// Computes `(a_i*x_i, b_i*x_i)` before advancing to the next coefficient.
    ///
    /// # Safety
    /// All three disjoint matrices contain exactly `transform_len` complete coefficients
    /// in the same ring. `leaf_budget` is the enclosing transform's admitted
    /// power of two. Scratch covers that budget after reduction to the active
    /// prefix's largest power of two.
    pub unsafe fn pointwise_multiply_pair_with_executor<E: ParallelExecutor>(
        matrices: [&mut [Limb]; 3],
        transform_len: usize,
        leaf_budget: NonZeroUsize,
        plan: &PointwiseMulPlan,
        executor: &E,
        scratch: &mut [Limb],
    ) {
        let mod_bits = plan.bits;
        let work = PairWork {
            bits: mod_bits,
            cl: SsaRing::coeff_limbs(mod_bits),
            leaf_scratch: plan.scratch_len,
        };
        let parallelism = executor.parallelism().get();
        let leaves = if leaf_budget.get() > 1
            && transform_len >= 2
            && SsaTransform::has_parallel_work(
                transform_len,
                SsaPlan::basecase_product_cost(work.cl.get()).saturating_mul(2),
                parallelism,
            ) {
            // SAFETY: transform_len>=2 gives a representable floor-power
            // exponent below usize::BITS, so this positive shift fits.
            let active_leaves = unsafe {
                let log = usize::BITS
                    .unchecked_sub(1)
                    .unchecked_sub(transform_len.leading_zeros());
                1_usize.unchecked_shl(log)
            };
            leaf_budget.get().min(active_leaves)
        } else {
            1
        };
        debug_assert!(
            scratch.len() >= work.leaf_scratch.get().saturating_mul(leaves),
            "paired pointwise execution has one complete arena per leaf"
        );
        // Dispatch once for the complete traversal. Each callback is specialized
        // to its concrete arithmetic strategy before entering coefficient loops.
        match &plan.strategy {
            PointwiseMulStrategy::Basecase(basecase) => {
                // SAFETY: matrices, leaf count, and scratch match the retained
                // fixed-width multiplication plan. The callback sees canonical
                // coefficients with zero guards and a disjoint product arena.
                unsafe {
                    work.run::<false, _, _>(
                        matrices,
                        transform_len,
                        leaves,
                        scratch,
                        executor,
                        &|left, right, _, arena| {
                            Self::fermat_basecase_mul_assign_left(
                                left, right, mod_bits, *basecase, arena,
                            );
                        },
                    );
                }
            }
            PointwiseMulStrategy::Negacyclic(factorized) => {
                // SAFETY: the factor plan matches this ring; result and arena
                // are disjoint complete partitions. The shared input is immutable.
                unsafe {
                    work.run::<true, _, _>(
                        matrices,
                        transform_len,
                        leaves,
                        scratch,
                        executor,
                        &|left, right, result, arena| {
                            factorized.mul_assign_left(left, right, result, arena);
                        },
                    );
                }
            }
            PointwiseMulStrategy::Transform(transform) => {
                // SAFETY: the nested plan and arena match this ring; canonical
                // inputs and nested workspace are complete and disjoint. Each
                // product consumes its left input before reconstruction, while
                // the immutable shared right input remains live for both calls.
                unsafe {
                    work.run::<false, _, _>(
                        matrices,
                        transform_len,
                        leaves,
                        scratch,
                        executor,
                        &|left, right, _, arena| {
                            transform.mul_assign_left(left, right, arena);
                        },
                    );
                }
            }
        }
    }
}

impl PairWork {
    /// Forks exact coefficient ranges, carrying the precomputed power-of-two
    /// leaf count instead of deriving a scheduling tree at each recursive node.
    ///
    /// # Safety
    /// All matrices contain exactly `count` complete coefficients; `leaves` is
    /// a positive power of two no greater than `count.max(1)`. Scratch contains
    /// that many private arenas. The callback accepts canonical ordinary
    /// coefficients and preserves its immutable shared input.
    unsafe fn run<const STAGED: bool, E, P>(
        &self,
        matrices: [&mut [Limb]; 3],
        count: usize,
        leaves: usize,
        scratch: &mut [Limb],
        executor: &E,
        product: &P,
    ) where
        E: ParallelExecutor,
        P: Fn(&mut [Limb], &[Limb], &mut [Limb], &mut [Limb]) + Sync,
    {
        if leaves == 1 {
            // SAFETY: this leaf owns complete matrices and one disjoint arena.
            unsafe {
                self.sequential::<STAGED, P>(matrices, count, scratch, product);
            }
            return;
        }
        let [a, b, x] = matrices;
        let half_count = count >> 1;
        // SAFETY: leaves>=2 and count>=leaves give positive child widths.
        let right_count = unsafe { count.unchecked_sub(half_count) };
        // SAFETY: half_count<=count, whose complete matrix span was admitted.
        let half_span = unsafe { half_count.unchecked_mul(self.cl.get()) };
        // SAFETY: the three initialized matrices have equal complete widths,
        // so the half coefficient boundary is valid in each disjoint range.
        let ((a_left, a_right), (b_left, b_right), (x_left, x_right)) = unsafe {
            (
                a.split_at_mut_unchecked(half_span),
                b.split_at_mut_unchecked(half_span),
                x.split_at_mut_unchecked(half_span),
            )
        };
        let half_leaves = leaves >> 1;
        // SAFETY: the validated full arena contains leaves complete workspaces,
        // so this prefix product is representable and within scratch.
        let half_scratch = unsafe { self.leaf_scratch.get().unchecked_mul(half_leaves) };
        // SAFETY: the checked full arena contains leaves private workspaces,
        // and half_leaves<=leaves bounds this disjoint partition.
        let (left_scratch, right_scratch) = unsafe { scratch.split_at_mut_unchecked(half_scratch) };
        let ((), ()) = executor.join(
            || {
                // SAFETY: the first halves own disjoint complete coefficient
                // ranges and half the private leaf arenas.
                unsafe {
                    self.run::<STAGED, E, P>(
                        [a_left, b_left, x_left],
                        half_count,
                        half_leaves,
                        left_scratch,
                        executor,
                        product,
                    );
                }
            },
            || {
                // SAFETY: the second halves own the remaining disjoint ranges
                // and arenas. Only the immutable strategy is shared.
                unsafe {
                    self.run::<STAGED, E, P>(
                        [a_right, b_right, x_right],
                        right_count,
                        half_leaves,
                        right_scratch,
                        executor,
                        product,
                    );
                }
            },
        );
    }

    /// Normalizes x once, then consumes it immutably for both adjacent products.
    ///
    /// # Safety
    /// The three matrices contain exactly `count` disjoint complete coefficients, and
    /// scratch holds the selected product arena. `STAGED` is true exactly for
    /// the negacyclic strategy, which needs a separate result coefficient.
    unsafe fn sequential<const STAGED: bool, P>(
        &self,
        matrices: [&mut [Limb]; 3],
        count: usize,
        scratch: &mut [Limb],
        product: &P,
    ) where
        P: Fn(&mut [Limb], &[Limb], &mut [Limb], &mut [Limb]),
    {
        let [a, b, x] = matrices;
        // SAFETY: STAGED reserves one complete result coefficient before the
        // product arena; un-staged strategies use the whole arena directly.
        let (result, arena) =
            unsafe { scratch.split_at_mut_unchecked(if STAGED { self.cl.get() } else { 0 }) };
        // SAFETY: an admitted positive ring has a data limb and its guard,
        // so cl>=2 and the initialized data prefix ends at cl-1.
        let data_len = unsafe { self.cl.get().unchecked_sub(1) };
        for index in 0..count {
            // SAFETY: the admitted count spans each initialized matrix exactly;
            // index<count addresses one complete coefficient in each disjoint range.
            let (left_a, left_b, shared) = unsafe {
                let offset = index.unchecked_mul(self.cl.get());
                let end = offset.unchecked_add(self.cl.get());
                (
                    a.get_unchecked_mut(offset..end),
                    b.get_unchecked_mut(offset..end),
                    x.get_unchecked_mut(offset..end),
                )
            };
            // SAFETY: all three spans are initialized complete coefficients;
            // normalization establishes their canonical guard representations.
            unsafe {
                let shared_neg_one = SsaRing::normalize(shared, self.bits);
                // An ordinary canonical zero annihilates both products. The
                // shared coefficient is scanned once; the left inputs need no
                // normalization because every output digit is overwritten.
                if !shared_neg_one
                    && shared
                        .get_unchecked(..data_len)
                        .iter()
                        .all(|&limb| limb == 0)
                {
                    left_a.fill(0);
                    left_b.fill(0);
                    continue;
                }
                let first_neg_one = SsaRing::normalize(left_a, self.bits);
                let second_neg_one = SsaRing::normalize(left_b, self.bits);
                if !SsaPointwise::write_pointwise_special_product(
                    left_a,
                    shared,
                    self.bits,
                    first_neg_one,
                    shared_neg_one,
                ) {
                    product(left_a, shared, result, arena);
                }
                if !SsaPointwise::write_pointwise_special_product(
                    left_b,
                    shared,
                    self.bits,
                    second_neg_one,
                    shared_neg_one,
                ) {
                    product(left_b, shared, result, arena);
                }
            }
        }
    }
}
