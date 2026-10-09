//! Fused row products against schoolbook multiplication and staged TFTs.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded test plans establish exact disjoint operand, matrix, output, and sentinel spans"
)]

use core::num::NonZeroUsize;

use alloc::vec;

use proptest::prelude::*;

use crate::{int::logic::unsigned::math::mul::Schoolbook, parallel::ParallelExecutor};

use super::super::{
    FftPlan, LIMB_BITS, Limb, MulTransformPlan, SsaPointwise, SsaTransform, TruncatedTransform,
};

struct WorkerBudget(NonZeroUsize);

impl ParallelExecutor for WorkerBudget {
    fn parallelism(&self) -> NonZeroUsize {
        self.0
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (left(), right())
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 8 }))]

    #[test]
    fn fused_products_preserve_supports_dirty_arenas_and_exact_outputs(
        size in prop::sample::select(if cfg!(miri) { vec![32_usize] } else { vec![32_usize, 128, 512, 2048] }),
        support_seed in any::<usize>(), split_seed in any::<usize>(), seed in any::<Limb>(),
    ) {
        let plan = MulTransformPlan::new(FftPlan::new(size * LIMB_BITS));
        let half = plan.transform_len >> 1;
        let support = half + 1 + support_seed.rem_euclid(half - 1);
        let active_left = 1 + split_seed.rem_euclid(support);
        for workers in if cfg!(miri) { &[1][..] } else { &[1, 2, 3, 5, 8][..] } {
            check_product(&plan, support, active_left, seed, &WorkerBudget(NonZeroUsize::new(*workers).expect("positive workers")));
        }
    }
}

#[cfg_attr(
    miri,
    ignore = "the native sweep checks every partial row at a 2048-limb ring; smaller arbitrary fused products run under Miri"
)]
#[test]
fn every_partial_row_and_non_power_of_two_worker_budget_matches_schoolbook() {
    let plan = MulTransformPlan::new(FftPlan::new(2048 * LIMB_BITS));
    let log = plan.transform_len.trailing_zeros();
    let width = 1_usize << (log - (log >> 1));
    let aligned = ((plan.transform_len * 3) >> 2).div_euclid(width) * width;
    for support in
        (aligned..aligned + width).chain([(plan.transform_len >> 1) + 1, plan.transform_len - 1])
    {
        for workers in [1, 2, 3, 5, 8] {
            for active_left in [1, (support >> 1) + 1, support] {
                check_product(
                    &plan,
                    support,
                    active_left,
                    Limb::MAX,
                    &WorkerBudget(NonZeroUsize::new(workers).expect("positive workers")),
                );
            }
        }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(
    miri,
    ignore = "native Rayon scheduling is checked with a three-worker pool; deterministic worker budgets cover the same partitions under Miri"
)]
#[test]
fn rayon_fused_rows_match_schoolbook_in_a_three_worker_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("test pool");
    let plan = MulTransformPlan::new(FftPlan::new(2048 * LIMB_BITS));
    pool.install(|| {
        crate::parallel::DefaultExecutor::with_resolved(|executor| {
            for support in [
                (plan.transform_len >> 1) + 1,
                (plan.transform_len * 3) >> 2,
                plan.transform_len - 1,
            ] {
                check_product(&plan, support, (support >> 1) + 1, 17, executor);
            }
        });
    });
}

fn check_product<E: ParallelExecutor>(
    plan: &MulTransformPlan,
    support: usize,
    active_left: usize,
    seed: Limb,
    executor: &E,
) {
    let active = [active_left, support + 1 - active_left];
    let inputs = active.map(|count| {
        let bits = count * plan.chunk_bits.get();
        let mut state = seed;
        let mut data: alloc::vec::Vec<Limb> = (0..bits.div_ceil(LIMB_BITS))
            .map(|_| {
                state = state.wrapping_mul(33).wrapping_add(7);
                state
            })
            .collect();
        let tail = data.last_mut().expect("positive support");
        let remainder = bits.rem_euclid(LIMB_BITS);
        if remainder != 0 {
            *tail &= Limb::MAX >> (LIMB_BITS - remainder);
        }
        *tail |= 1;
        data
    });
    let output_len = plan.modulus_bits.div_euclid(LIMB_BITS) + 1;
    let mut expected = vec![0; output_len];
    let product_len = inputs[0].len() + inputs[1].len();
    Schoolbook::mul(&mut expected[..product_len], &inputs[0], &inputs[1]);
    let mut transform = TruncatedTransform::new(plan);
    transform.max_resident = 0;
    let workers = executor.parallelism().get();
    let scratch_len = plan.transform_mul_scratch(workers);
    let mut fused_scratch = vec![Limb::MAX; scratch_len + 2];
    let mut staged_scratch = fused_scratch.clone();
    let scratch_address = fused_scratch.as_ptr();
    let mut fused = vec![37; output_len + 2];
    let mut staged = fused.clone();
    for dirty in [Limb::MAX, 21] {
        fused_scratch[1..=scratch_len].fill(dirty);
        staged_scratch[1..=scratch_len].fill(dirty);
        fused[1..=output_len].fill(dirty);
        staged[1..=output_len].fill(dirty);
        // SAFETY: active supports sum to support+1<=K, hence the exact product
        // is below 2^modulus_bits and has a zero time tail. The initialized
        // arenas have their exact worker-sized plan spans; all buffers are
        // disjoint and both outputs include the complete guarded ring span.
        unsafe {
            SsaTransform::fused_matrix_truncated_mul(
                &inputs[0],
                &inputs[1],
                &mut fused[1..=output_len],
                active[0],
                active[1],
                support,
                plan,
                &transform,
                executor,
                &mut fused_scratch[1..=scratch_len],
            );
            SsaTransform::truncated_product(
                [&inputs[0][..], &inputs[1][..]],
                [&mut staged[1..=output_len]],
                active,
                [support; 2],
                [support],
                plan,
                executor,
                &mut staged_scratch[1..=scratch_len],
                |[left, right], work| {
                    SsaPointwise::pointwise_multiply_with_executor(
                        left,
                        right,
                        support,
                        plan.pointwise_leaf_count(workers),
                        plan.pointwise(),
                        executor,
                        work,
                    );
                },
            );
        }
        assert_eq!(
            &fused[1..=output_len],
            expected,
            "support={support}, active={active:?}, workers={workers}"
        );
        assert_eq!(&staged[1..=output_len], expected);
        assert_eq!(
            (
                fused[0],
                fused[output_len + 1],
                staged[0],
                staged[output_len + 1]
            ),
            (37, 37, 37, 37)
        );
        assert_eq!(
            (
                fused_scratch[0],
                fused_scratch[scratch_len + 1],
                staged_scratch[0],
                staged_scratch[scratch_len + 1]
            ),
            (Limb::MAX, Limb::MAX, Limb::MAX, Limb::MAX)
        );
        assert_eq!(fused_scratch.as_ptr(), scratch_address);
    }
}
