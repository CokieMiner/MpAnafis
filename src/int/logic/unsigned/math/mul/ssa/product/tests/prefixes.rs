//! Pointwise prefix alignment, leaf budgets, and private execution arenas.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded spectral counts provide complete coefficients and exact leaf arenas with external sentinels"
)]

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use alloc::{vec, vec::Vec};

use crate::parallel::{ParallelExecutor, SequentialExecutor};

use super::super::{
    super::tests::CountingExecutor, FftPlan, LIMB_BITS, Limb, PointwiseMulPlan,
    PointwiseSquarePlan, SsaPointwise, SsaRing,
};

struct PrefixExecutor {
    workers: NonZeroUsize,
    joins: AtomicUsize,
}

impl ParallelExecutor for PrefixExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        self.workers
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _previous = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

fn coefficients(count: usize, bits: usize, seed: Limb) -> Vec<Limb> {
    let cl = SsaRing::coeff_limbs(bits).get();
    let mut result = vec![0; count * cl];
    let mut state = seed;
    for slot in result.chunks_exact_mut(cl) {
        for limb in slot.iter_mut().take(cl - 1) {
            state = state.wrapping_mul(33).wrapping_add(7);
            *limb = state;
        }
    }
    result
}

#[test]
#[cfg_attr(
    miri,
    ignore = "This sweep uses 8192-bit nested products; smaller non-power-of-two prefixes retain the arena and fork checks under Miri"
)]
fn odd_prefixes_preserve_coefficient_and_scratch_boundaries() {
    let bits = 8192;
    let cl = SsaRing::coeff_limbs(bits).get();
    let mul = PointwiseMulPlan::from(bits);
    let sqr = PointwiseSquarePlan::from(bits);
    let executor = CountingExecutor::default();
    for count in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
        check_pointwise_prefix(count, bits, cl, &mul, &sqr, &executor);
    }
    assert!(
        executor.joins.load(Ordering::Relaxed) > 0,
        "wide pointwise prefixes exercise parallel splitting"
    );
}

#[test]
fn pointwise_leaf_budget_is_the_smallest_complete_balanced_partition() {
    let ring_widths: &[usize] = if cfg!(miri) {
        &[2 * LIMB_BITS, 33 * LIMB_BITS]
    } else {
        &[LIMB_BITS, 2 * LIMB_BITS, 512, 8192, 32_768, 131_072]
    };
    for &bits in ring_widths {
        let plan = FftPlan::new(bits);
        let len = plan.transform_len;
        for requested in (0..=len + 1).chain([usize::MAX]) {
            let workers = requested.max(1).min(len);
            let maximum_leaf_len = len.div_ceil(workers);
            let mut leaves = 1;
            // Enumerate balanced binary partitions independently of the
            // production logarithm: the longest leaf holds ceil(len/leaves).
            while len.div_ceil(leaves) > maximum_leaf_len {
                leaves *= 2;
            }
            assert_eq!(
                plan.pointwise_leaf_count(requested).get(),
                leaves,
                "K={len}, requested workers={requested} admits the minimal complete partition"
            );
        }
    }
}

#[test]
fn truncated_pointwise_ranges_preserve_the_full_plan_leaf_budget() {
    // Miri retains the over-admission shape with basecase coefficients;
    // native execution additionally covers recursively transformed products.
    let bits = if cfg!(miri) { 33 * LIMB_BITS } else { 8192 };
    let counts: &[usize] = if cfg!(miri) { &[17] } else { &[17, 18, 33, 34] };
    let worker_counts: &[usize] = if cfg!(miri) { &[9] } else { &[9, 17] };
    let cl = SsaRing::coeff_limbs(bits).get();
    let mul = PointwiseMulPlan::from(bits);
    let sqr = PointwiseSquarePlan::from(bits);
    for &workers in worker_counts {
        let executor = PrefixExecutor {
            workers: NonZeroUsize::new(workers).expect("positive test worker budget"),
            joins: AtomicUsize::new(0),
        };
        for &count in counts {
            check_pointwise_prefix(count, bits, cl, &mul, &sqr, &executor);
        }
        assert!(
            executor.joins.load(Ordering::Relaxed) > 0,
            "prefix work must exercise the admitted parallel partitions"
        );
    }
}

/// Runs one odd frequency count through paired, squared, and single products,
/// checking coefficient alignment and scratch sentinels on every path.
fn check_pointwise_prefix<E: ParallelExecutor>(
    count: usize,
    bits: usize,
    cl: usize,
    mul: &PointwiseMulPlan,
    sqr: &PointwiseSquarePlan,
    executor: &E,
) {
    let workers = executor.parallelism().get();
    let leaf_budget = FftPlan::new(bits).pointwise_leaf_count(workers);
    let width = count * cl;
    let source_a = coefficients(count, bits, 43);
    let source_b = coefficients(count, bits, 47);
    let source_x = coefficients(count, bits, 53);
    let mut expected_a = source_a.clone();
    let mut expected_b = source_b.clone();
    let mut expected_x = source_x.clone();
    let mut expected_square = source_a.clone();
    let mut sequential = vec![Limb::MAX; mul.scratch_len.get().max(sqr.scratch_len.get())];
    // SAFETY: complete disjoint count-coefficient spectra and exactly one
    // coefficient-plan arena; the sequential reference uses all count slots.
    unsafe {
        SsaPointwise::pointwise_multiply_pair_with_executor(
            [&mut expected_a, &mut expected_b, &mut expected_x],
            count,
            NonZeroUsize::MIN,
            mul,
            &SequentialExecutor,
            &mut sequential,
        );
        SsaPointwise::pointwise_square_with_executor(
            &mut expected_square,
            count,
            NonZeroUsize::MIN,
            sqr,
            &SequentialExecutor,
            &mut sequential,
        );
    }
    let leaves = leaf_budget.get().min(1 << count.ilog2());
    assert!(leaves.is_power_of_two() && leaves <= count);
    let mut arena = vec![Limb::MAX; leaves * mul.scratch_len.get() + 2];
    let mut square_arena = vec![Limb::MAX; leaves * sqr.scratch_len.get() + 2];
    let mut actual_a = vec![Limb::MAX; width + 2];
    let mut actual_b = actual_a.clone();
    let mut actual_x = actual_a.clone();
    actual_a[1..=width].copy_from_slice(&source_a);
    actual_b[1..=width].copy_from_slice(&source_b);
    actual_x[1..=width].copy_from_slice(&source_x);
    let mut actual_square = actual_a.clone();
    let work_end = arena.len() - 1;
    let square_work_end = square_arena.len() - 1;
    // SAFETY: bounded windows contain exact complete coefficient counts and
    // leaf arenas with an external sentinel at each end. Splits must remain
    // coefficient aligned even when both count and cl are odd.
    unsafe {
        SsaPointwise::pointwise_multiply_pair_with_executor(
            [
                &mut actual_a[1..=width],
                &mut actual_b[1..=width],
                &mut actual_x[1..=width],
            ],
            count,
            leaf_budget,
            mul,
            executor,
            &mut arena[1..work_end],
        );
        SsaPointwise::pointwise_square_with_executor(
            &mut actual_square[1..=width],
            count,
            leaf_budget,
            sqr,
            executor,
            &mut square_arena[1..square_work_end],
        );
    }
    assert_eq!(&actual_a[1..=width], expected_a, "paired A count={count}");
    assert_eq!(&actual_b[1..=width], expected_b, "paired B count={count}");
    assert_eq!(
        &actual_square[1..=width],
        expected_square,
        "square count={count}"
    );
    actual_a[1..=width].copy_from_slice(&source_a);
    actual_x[1..=width].copy_from_slice(&source_x);
    // SAFETY: the same complete windows and exact leaf arena also cover
    // ordinary multiplication, whose split carries the explicit count.
    unsafe {
        SsaPointwise::pointwise_multiply_with_executor(
            &mut actual_a[1..=width],
            &mut actual_x[1..=width],
            count,
            leaf_budget,
            mul,
            executor,
            &mut arena[1..work_end],
        );
    }
    assert_eq!(&actual_a[1..=width], expected_a, "single count={count}");
    for output in [
        &actual_a,
        &actual_b,
        &actual_x,
        &actual_square,
        &arena,
        &square_arena,
    ] {
        assert_eq!(output.first(), Some(&Limb::MAX));
        assert_eq!(output.last(), Some(&Limb::MAX));
    }
}
