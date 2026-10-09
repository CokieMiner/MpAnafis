//! Deterministic executors that count joins without creating worker threads.

use super::*;

#[derive(Debug, Default)]
pub struct CountingExecutor {
    pub joins: AtomicUsize,
}

impl ParallelExecutor for CountingExecutor {
    fn parallelism(&self) -> NonZeroUsize {
        NonZeroUsize::new(8).expect("eight logical workers")
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _ = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

#[derive(Debug, Default)]
pub struct OneSlotCountingExecutor {
    pub joins: AtomicUsize,
}

impl ParallelExecutor for OneSlotCountingExecutor {
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
        let _previous = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}
