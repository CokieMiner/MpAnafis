//! Internal synchronous execution adapters for independent arithmetic work.
//!
//! `MpAnafis` never creates or accepts a thread pool. With the `rayon` feature,
//! arithmetic uses Rayon's current pool (or its global pool); without it, the
//! same fork/join surface executes sequentially.

use core::num::NonZeroUsize;

#[cfg(feature = "rayon")]
use rayon::{current_num_threads, join};

/// Schedules independent synchronous arithmetic work.
///
/// This trait is internal implementation plumbing. Keeping the two built-in
/// implementations behind one generic surface lets the no-Rayon build compile
/// the same kernels without exposing a user-defined scheduler contract.
pub trait ParallelExecutor: Sync {
    /// Returns the scheduling width used to size one operation's scratch.
    fn parallelism(&self) -> NonZeroUsize;

    /// Runs two independent closures and returns after both have quiesced.
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;
}

/// Internal executor for no-Rayon builds and forced sequential tuner rows.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SequentialExecutor;

impl ParallelExecutor for SequentialExecutor {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    #[inline]
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

/// Internal adapter for Rayon's current or global work-stealing pool.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct RayonExecutor;

#[cfg(feature = "rayon")]
impl ParallelExecutor for RayonExecutor {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(current_num_threads()).unwrap_or(NonZeroUsize::MIN)
    }

    #[inline]
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

/// Executor view retaining the width recorded by a prepared tuner plan.
#[cfg(any(feature = "rayon", feature = "_internal-tune"))]
#[derive(Clone, Copy, Debug)]
pub struct FixedParallelismExecutor<'executor, E: ?Sized> {
    executor: &'executor E,
    parallelism: NonZeroUsize,
}

#[cfg(any(feature = "rayon", feature = "_internal-tune"))]
impl<'executor, E: ParallelExecutor + ?Sized> FixedParallelismExecutor<'executor, E> {
    /// Borrows `executor` with the logical width used to size prepared scratch.
    #[must_use]
    pub const fn new(executor: &'executor E, parallelism: NonZeroUsize) -> Self {
        Self {
            executor,
            parallelism,
        }
    }
}

#[cfg(any(feature = "rayon", feature = "_internal-tune"))]
impl<E: ParallelExecutor + ?Sized> ParallelExecutor for FixedParallelismExecutor<'_, E> {
    #[inline]
    fn parallelism(&self) -> NonZeroUsize {
        self.parallelism
    }

    #[inline]
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        if self.parallelism.get() == 1 {
            (left(), right())
        } else {
            self.executor.join(left, right)
        }
    }
}

/// Feature-selected execution boundary used by ordinary arithmetic.
pub struct DefaultExecutor;

impl DefaultExecutor {
    /// Resolves the built-in backend once for a complete arithmetic operation.
    #[cfg(feature = "rayon")]
    pub fn with_resolved<R>(
        action: impl FnOnce(&FixedParallelismExecutor<'_, RayonExecutor>) -> R,
    ) -> R {
        let executor = RayonExecutor;
        let resolved = FixedParallelismExecutor::new(&executor, executor.parallelism());
        action(&resolved)
    }

    /// Resolves the sequential backend when Rayon is not compiled in.
    #[cfg(not(feature = "rayon"))]
    pub fn with_resolved<R>(action: impl FnOnce(&SequentialExecutor) -> R) -> R {
        action(&SequentialExecutor)
    }
}
