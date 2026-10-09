//! Benchmark cases with explicit operand dimensions and worker budgets.

use core::fmt;

/// One balanced width measured at a stated worker budget.
#[derive(Clone, Copy, Debug)]
pub struct WorkerCase {
    pub len: usize,
    pub workers: usize,
}

impl fmt::Display for WorkerCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}-limbs/{}-workers", self.len, self.workers)
    }
}

/// One operand shape measured at a stated worker budget.
#[derive(Clone, Copy, Debug)]
pub struct ShapeWorkerCase {
    pub larger_len: usize,
    pub smaller_len: usize,
    pub workers: usize,
}

impl fmt::Display for ShapeWorkerCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}x{}-limbs/{}-workers",
            self.larger_len, self.smaller_len, self.workers
        )
    }
}

pub fn parallel_worker_cases<const N: usize>(sizes: [usize; N]) -> Vec<WorkerCase> {
    let workers = ambient_workers();
    if workers <= 1 {
        return Vec::new();
    }
    worker_cases(sizes, workers)
}

pub fn parallel_shape_cases<const N: usize>(shapes: [(usize, usize); N]) -> Vec<ShapeWorkerCase> {
    let workers = ambient_workers();
    if workers <= 1 {
        return Vec::new();
    }
    shape_cases(shapes, workers)
}

pub fn worker_cases<const N: usize>(sizes: [usize; N], workers: usize) -> Vec<WorkerCase> {
    sizes
        .into_iter()
        .map(|len| WorkerCase { len, workers })
        .collect()
}

pub fn shape_cases<const N: usize>(
    shapes: [(usize, usize); N],
    workers: usize,
) -> Vec<ShapeWorkerCase> {
    shapes
        .into_iter()
        .map(|(larger_len, smaller_len)| ShapeWorkerCase {
            larger_len,
            smaller_len,
            workers,
        })
        .collect()
}

#[cfg_attr(
    not(feature = "rayon"),
    expect(
        clippy::missing_const_for_fn,
        reason = "calls non-const rayon::current_num_threads when rayon is active"
    )
)]
pub fn ambient_workers() -> usize {
    #[cfg(feature = "rayon")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "rayon"))]
    {
        1
    }
}
