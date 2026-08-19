//! Parallel executor regression tests.

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "rayon")]
use core::panic::AssertUnwindSafe;
#[cfg(feature = "rayon")]
use core::sync::atomic::AtomicBool;

#[cfg(feature = "rayon")]
use alloc::{boxed::Box, sync::Arc};

#[cfg(feature = "rayon")]
use std::panic::{catch_unwind, resume_unwind};

#[cfg(feature = "rayon")]
use super::RayonExecutor;
use super::{ParallelExecutor, SequentialExecutor};

#[derive(Debug)]
struct CountingExecutor {
    joins: AtomicUsize,
}

impl CountingExecutor {
    const fn new() -> Self {
        Self {
            joins: AtomicUsize::new(0),
        }
    }
}

impl ParallelExecutor for CountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).expect("two is nonzero")
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _previous_join_count = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

#[test]
fn sequential_executor_runs_borrowed_disjoint_slices() {
    let executor = SequentialExecutor::default();
    let mut values = [1_u32, 2, 3, 4];
    let (left, right) = values.split_at_mut(2);

    let (left_sum, right_sum) = executor.join(
        || {
            let (left_first, left_second) = left.split_at_mut(1);
            let first_value = left_first.first_mut().expect("left range has one element");
            let second_value = left_second
                .first()
                .copied()
                .expect("left range has two elements");
            *first_value += second_value;
            *first_value
        },
        || {
            let (right_first, right_second) = right.split_at_mut(1);
            let first_value = right_first
                .first_mut()
                .expect("right range has one element");
            let second_value = right_second
                .first()
                .copied()
                .expect("right range has two elements");
            *first_value += second_value;
            *first_value
        },
    );

    assert_eq!(left_sum, 3, "left borrowed range must be updated");
    assert_eq!(right_sum, 7, "right borrowed range must be updated");
    assert_eq!(values, [3, 2, 7, 4], "both disjoint ranges must persist");
}

#[test]
fn custom_executor_is_usable_without_rayon() {
    let executor = CountingExecutor::new();
    let (left, right) = executor.join(
        || {
            let (first, second) = executor.join(|| 8_u32, || 12_u32);
            first + second
        },
        || {
            let (first, second) = executor.join(|| 10_u32, || 12_u32);
            first + second
        },
    );

    assert_eq!(
        (left, right),
        (20, 22),
        "custom join must return both values"
    );
    assert_eq!(
        executor.joins.load(Ordering::Relaxed),
        3,
        "the outer and both nested joins must be observed"
    );
    assert_eq!(
        executor.parallelism().get(),
        2,
        "parallelism is a scheduling hint"
    );
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_executor_reuses_the_current_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test Rayon pool must build");

    pool.install(|| {
        let executor = RayonExecutor::default();
        assert_eq!(
            executor.parallelism().get(),
            2,
            "adapter must observe the installed pool"
        );
        let (left_threads, right_threads) =
            executor.join(rayon::current_num_threads, rayon::current_num_threads);
        assert_eq!(left_threads, 2, "left task must use the installed pool");
        assert_eq!(right_threads, 2, "right task must use the installed pool");
    });
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_executor_quiesces_the_other_branch_before_propagating_panic() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test Rayon pool must build");
    let completed = Arc::new(AtomicBool::new(false));
    let right_completed = Arc::clone(&completed);

    let result = catch_unwind(AssertUnwindSafe(|| {
        pool.install(|| {
            let executor = RayonExecutor::default();
            executor.join(
                || resume_unwind(Box::new("intentional executor-contract panic")),
                || right_completed.store(true, Ordering::Release),
            )
        })
    }));

    assert!(result.is_err(), "Rayon must propagate a branch panic");
    assert!(
        completed.load(Ordering::Acquire),
        "Rayon must finish already-started borrowed work before unwinding join"
    );
}
