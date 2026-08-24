//! Benchmark cases that make the ambient executor budget part of every label.

use core::fmt;

use mp_anafis::tune_api::tier::transform::TransformExecutor;

/// One balanced width measured at the ambient executor's worker budget.
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

/// One operand shape measured at the ambient executor's worker budget.
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

pub fn ambient_worker_cases<const N: usize>(sizes: [usize; N]) -> Vec<WorkerCase> {
    let workers = ambient_workers();
    sizes
        .into_iter()
        .map(|len| WorkerCase { len, workers })
        .collect()
}

pub fn parallel_worker_cases<const N: usize>(sizes: [usize; N]) -> Vec<WorkerCase> {
    if ambient_workers() <= 1 {
        return Vec::new();
    }
    ambient_worker_cases(sizes)
}

pub fn ambient_shape_cases<const N: usize>(shapes: [(usize, usize); N]) -> Vec<ShapeWorkerCase> {
    let workers = ambient_workers();
    shapes
        .into_iter()
        .map(|(larger_len, smaller_len)| ShapeWorkerCase {
            larger_len,
            smaller_len,
            workers,
        })
        .collect()
}

pub fn parallel_shape_cases<const N: usize>(shapes: [(usize, usize); N]) -> Vec<ShapeWorkerCase> {
    if ambient_workers() <= 1 {
        return Vec::new();
    }
    ambient_shape_cases(shapes)
}

fn ambient_workers() -> usize {
    TransformExecutor::Default.parallelism().get()
}
