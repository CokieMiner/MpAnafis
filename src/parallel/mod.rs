//! Internal execution policy for independent arithmetic work.
//!
//! The `rayon` feature uses the application's current or global Rayon pool.
//! `MpAnafis` owns no pool and accepts no user-defined executor. Builds without
//! Rayon execute the identical algorithm graph on the calling thread.

mod api;
#[cfg(test)]
mod tests;

#[cfg(feature = "_internal-tune")]
pub use api::FixedParallelismExecutor;
#[cfg(any(test, feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use api::SequentialExecutor;
pub use api::{DefaultExecutor, ParallelExecutor};
