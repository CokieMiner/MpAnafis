//! Built-in execution-policy regression tests.

#[cfg(feature = "rayon")]
use core::{
    panic::AssertUnwindSafe,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "rayon")]
use alloc::{boxed::Box, sync::Arc};
#[cfg(feature = "rayon")]
use std::panic::{catch_unwind, resume_unwind};

#[cfg(feature = "rayon")]
use super::api::RayonExecutor;
use super::{DefaultExecutor, ParallelExecutor, SequentialExecutor};

#[test]
fn sequential_executor_runs_borrowed_disjoint_slices() {
    let mut values = [1_u32, 2, 3, 4];
    let (left, right) = values.split_at_mut(2);

    let (left_sum, right_sum) = SequentialExecutor.join(
        || {
            let sum = left.iter().copied().sum();
            if let Some(first) = left.first_mut() {
                *first = sum;
            }
            sum
        },
        || {
            let sum = right.iter().copied().sum();
            if let Some(first) = right.first_mut() {
                *first = sum;
            }
            sum
        },
    );

    assert_eq!((left_sum, right_sum), (3, 7));
    assert_eq!(values, [3, 2, 7, 4]);
}

#[cfg(not(feature = "rayon"))]
#[test]
fn default_executor_is_sequential_without_rayon() {
    DefaultExecutor::with_resolved(|executor| {
        assert_eq!(executor.parallelism().get(), 1);
        assert_eq!(executor.join(|| 4_u32, || 6_u32), (4, 6));
    });
}

#[cfg(feature = "rayon")]
#[test]
fn active_rayon_pool_controls_default_executor() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test Rayon pool must build");

    pool.install(|| {
        DefaultExecutor::with_resolved(|executor| {
            assert_eq!(executor.parallelism().get(), 2);
            let (left, right) =
                executor.join(rayon::current_num_threads, rayon::current_num_threads);
            assert_eq!((left, right), (2, 2));
        });
    });
}

#[cfg(feature = "rayon")]
#[test]
fn one_thread_rayon_pool_forces_sequential_width() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("test Rayon pool must build");

    pool.install(|| {
        DefaultExecutor::with_resolved(|executor| assert_eq!(executor.parallelism().get(), 1));
    });
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_join_quiesces_before_propagating_panic() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test Rayon pool must build");
    let completed = Arc::new(AtomicBool::new(false));
    let right_completed = Arc::clone(&completed);

    let result = catch_unwind(AssertUnwindSafe(|| {
        pool.install(|| {
            RayonExecutor.join(
                || resume_unwind(Box::new("intentional executor-contract panic")),
                || right_completed.store(true, Ordering::Release),
            )
        })
    }));

    assert!(result.is_err(), "Rayon must propagate a branch panic");
    assert!(
        completed.load(Ordering::Acquire),
        "Rayon must quiesce already-started borrowed work"
    );
}
