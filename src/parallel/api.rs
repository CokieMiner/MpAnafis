//! Internal synchronous execution adapters for independent arithmetic work.
//!
//! Adapters report a planning budget and join independent closures before
//! returning their results. The default backend uses the active Rayon pool
//! when `rayon` is enabled and sequential execution otherwise.

use core::num::NonZeroUsize;
#[cfg(feature = "rayon")]
use std::sync::OnceLock;

#[cfg(feature = "rayon")]
use rayon::{current_num_threads, current_thread_index, join};

#[cfg(feature = "rayon")]
use super::narrow_default_pool;

/// Schedules independent synchronous arithmetic work.
///
/// Implementations provide sequential or Rayon execution with the same join
/// interface.
pub trait ParallelExecutor: Sync {
    /// Returns the maximum concurrency degree used to size scratch buffers.
    ///
    /// The value is a nonzero planning budget. Algorithms partition this budget
    /// when subdividing scratch; the adapter does not reserve worker threads.
    fn parallelism(&self) -> NonZeroUsize;

    /// Executes two independent closures and returns their results.
    ///
    /// # Panics
    ///
    /// Propagates a closure panic after all started work has stopped. A
    /// sequential adapter does not start `right` if `left` panics.
    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send;
}

/// Resolves the default execution backend for an arithmetic operation.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct DefaultExecutor;

/// Executes closures sequentially on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SequentialExecutor;

/// Executes closures in the active Rayon pool.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct RayonExecutor;

/// Reports a fixed planning budget over a borrowed executor.
///
/// A budget of one executes joins sequentially. Larger budgets forward joins
/// to the underlying executor without reserving or creating workers.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy, Debug)]
pub struct FixedParallelismExecutor<'executor, E: ?Sized> {
    executor: &'executor E,
    parallelism: NonZeroUsize,
}

impl DefaultExecutor {
    /// Resolves the built-in backend once for a complete arithmetic operation.
    #[cfg(feature = "rayon")]
    pub fn with_resolved<R>(
        action: impl FnOnce(&FixedParallelismExecutor<'_, RayonExecutor>) -> R,
    ) -> R {
        static GLOBAL_PARALLELISM: OnceLock<NonZeroUsize> = OnceLock::new();
        let executor = RayonExecutor;
        // Calls inside Rayon workers use the active pool without initializing
        // the global pool policy.
        let parallelism = if current_thread_index().is_some() {
            executor.parallelism()
        } else {
            // The global pool's width is immutable after initialization.
            *GLOBAL_PARALLELISM.get_or_init(|| {
                narrow_default_pool();
                executor.parallelism()
            })
        };
        let resolved = FixedParallelismExecutor::new(&executor, parallelism);
        action(&resolved)
    }

    /// Resolves the sequential backend when Rayon is not compiled in.
    #[cfg(not(feature = "rayon"))]
    pub fn with_resolved<R>(action: impl FnOnce(&SequentialExecutor) -> R) -> R {
        action(&SequentialExecutor)
    }

    /// Resolves the built-in backend once at an explicit worker budget.
    ///
    /// Prepared transform plans use the supplied budget for scratch partitioning.
    /// Resolving this adapter does not initialize a worker pool.
    #[cfg(all(feature = "rayon", feature = "_internal-tune"))]
    pub fn with_resolved_parallelism<R>(
        parallelism: NonZeroUsize,
        action: impl FnOnce(&FixedParallelismExecutor<'_, RayonExecutor>) -> R,
    ) -> R {
        let executor = RayonExecutor;
        let resolved = FixedParallelismExecutor::new(&executor, parallelism);
        action(&resolved)
    }

    /// Resolves the sequential backend at an explicit worker budget.
    ///
    /// The sequential backend reports one worker regardless of the requested
    /// budget. Prepared plans validate that reported width during construction.
    #[cfg(all(not(feature = "rayon"), feature = "_internal-tune"))]
    pub fn with_resolved_parallelism<R>(
        _parallelism: NonZeroUsize,
        action: impl FnOnce(&SequentialExecutor) -> R,
    ) -> R {
        action(&SequentialExecutor)
    }
}

#[cfg(feature = "rayon")]
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

#[cfg(feature = "rayon")]
impl ParallelExecutor for RayonExecutor {
    /// Returns the ambient pool width used to size scratch buffers.
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

#[cfg(feature = "rayon")]
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
