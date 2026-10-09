//! Blocked products, worker partitions, and workspace sizing.

mod batches;
mod products;
mod sizing;

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
mod parallel;
