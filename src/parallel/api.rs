//! Synchronous execution adapters for independent arithmetic work.
//!
//! The executor abstraction is deliberately smaller than a thread-pool API:
//! arithmetic kernels need one synchronous fork/join operation and must be
//! able to borrow disjoint caller-owned buffers. An executor may run both
//! closures sequentially, and `parallelism` is only a scheduling hint; it does
//! not authorize an executor to create that many threads.

use core::num::NonZeroUsize;

#[cfg(feature = "rayon")]
use rayon::{current_num_threads, join};

/// Schedules independent synchronous work for arithmetic algorithms.
///
/// Implementations must not return from [`Self::join`] while either closure can
/// still access borrowed state. A normal return means both closures completed.
/// If either closure panics, an implementation must first quiesce any work it
/// started and then propagate the panic (or apply an equivalent documented
/// policy). Transform recursion may call `join` again from either closure, so
/// implementations must support nested joins without waiting on their own
/// worker in a way that can deadlock. The method is generic rather than
/// object-safe so callers can safely borrow disjoint slices without requiring
/// `'static` closures.
pub trait ParallelExecutor: Sync {
    /// Returns a nonzero scheduling hint for this executor.
    ///
    /// This value is not a thread-creation request. An executor may use its own
    /// affinity, first-touch, NUMA, or work-stealing policy when scheduling.
    #[must_use]
    fn parallelism(&self) -> NonZeroUsize;

    /// Runs two independent closures and returns both results after quiescence.
    ///
    /// The closures may borrow disjoint caller-owned data for the duration of
    /// this call. Implementations may run them sequentially. Panic handling is
    /// part of the executor contract: no started closure may retain access to
    /// borrowed state after this method propagates a panic. Either closure may
    /// recursively invoke this executor.
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;
}

/// A synchronous executor that never creates worker threads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct SequentialExecutor;

impl ParallelExecutor for SequentialExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let left_result = left();
        let right_result = right();
        (left_result, right_result)
    }
}

/// Adapter for Rayon’s current/global work-stealing pool.
///
/// This adapter does not create an `MpAnafis` pool. When called from a Rayon
/// `install` closure, Rayon schedules nested work on that pool; otherwise the
/// Rayon global pool is used according to Rayon’s normal policy.
#[cfg(feature = "rayon")]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct RayonExecutor;

#[cfg(feature = "rayon")]
impl ParallelExecutor for RayonExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(current_num_threads()).unwrap_or(NonZeroUsize::MIN)
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        join(left, right)
    }
}

/// The feature-selected executor used by parallel arithmetic APIs.
///
/// It is sequential for the zero-dependency baseline and for `std` alone. The
/// `rayon` feature selects [`RayonExecutor`] without creating a second pool.
#[cfg(feature = "rayon")]
pub type DefaultExecutor = RayonExecutor;

/// The feature-selected executor used by parallel arithmetic APIs.
///
/// This alias selects the zero-dependency sequential implementation unless the
/// optional `rayon` feature is enabled.
#[cfg(not(feature = "rayon"))]
pub type DefaultExecutor = SequentialExecutor;
