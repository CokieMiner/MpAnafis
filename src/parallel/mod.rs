//! Internal execution policy for independent arithmetic work.
//!
//! With `rayon`, independent subtasks use the active Rayon pool. Otherwise,
//! subtasks execute sequentially on the calling thread.

mod api;
#[cfg(feature = "rayon")]
mod topology;

pub use api::{DefaultExecutor, ParallelExecutor, SequentialExecutor};
#[cfg(feature = "rayon")]
pub use topology::narrow_default_pool;

#[cfg(test)]
mod tests;
