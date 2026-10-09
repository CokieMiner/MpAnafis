//! Scratch ownership, bucket state transitions, and thread-local reuse.

#[cfg(feature = "std")]
use super::BucketSlot;
use super::ScratchBuffer;

#[cfg(feature = "std")]
mod bucket;
#[cfg(feature = "std")]
mod pooling;
mod scratch;
