//! Execution-policy facade for independent arithmetic work.

mod api;
#[cfg(test)]
mod tests;

#[cfg(feature = "rayon")]
pub use api::RayonExecutor;
pub use api::{DefaultExecutor, ParallelExecutor, SequentialExecutor};
