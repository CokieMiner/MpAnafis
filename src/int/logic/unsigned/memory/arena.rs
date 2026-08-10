//! Thread-local scratch-buffer arena for reusable unsigned arithmetic storage.

#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use core::cmp::min;
use core::{
    mem::swap,
    ops::{Deref, DerefMut},
};
#[cfg(feature = "std")]
use std::thread_local;

use alloc::vec::Vec;

use super::Limb;

// ---------------------------------------------------------------------------
// Bucketed arena: instead of a flat Vec<Vec<Limb>>, we keep BUCKET_COUNT
// per-bucket Vecs.  Each bucket i holds buffers whose capacity is in
// [2^i, 2^(i+1)).  Lookup is O(1) instead of O(n).
// ---------------------------------------------------------------------------

/// Number of size buckets (covers capacities up to `2^BUCKET_COUNT` limbs).
#[cfg(feature = "std")]
const BUCKET_COUNT: usize = 24;

/// Maximum number of buffers retained per bucket to bound memory usage.
#[cfg(feature = "std")]
const MAX_PER_BUCKET: usize = 8;

/// Buffers with capacity below this threshold are dropped immediately
/// instead of being returned to the arena, avoiding TLS + `RefCell` overhead
/// for trivially re-allocatable small buffers.
/// TODO(tuning): benchmark this cutoff across allocator implementations and
/// operand-size distributions before changing it.
#[cfg(feature = "std")]
const SMALL_BUFFER_DROP_THRESHOLD: usize = 64;

/// Returns the bucket index for a given capacity.
/// Bucket 0 covers cap = 0..1, bucket 1 covers 1..2, bucket 2 covers 2..4, etc.
///
/// Uses a branchless approach: `cap.ilog2()` would panic on 0, so we OR with 1
/// to floor cap at 1 (ilog2(1) = 0, mapping cap=0 to bucket 0 as required).
#[cfg(feature = "std")]
#[inline(always)]
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "ilog2 returns u32 which always fits in usize even on 16-bit targets; inlining eliminates function call overhead in hot allocator paths."
)]
fn bucket_for_capacity(cap: usize) -> usize {
    // Branchless: OR with 1 maps cap=0 to ilog2(1)=0, cap=1 to ilog2(1)=0, etc.
    // This avoids a separate branch for the cap == 0 case.
    let idx = cap.max(1).ilog2() as usize;
    min(idx, BUCKET_COUNT - 1)
}

#[cfg(feature = "std")]
struct BucketSlot {
    buffers: [Option<Vec<Limb>>; MAX_PER_BUCKET],
    len: usize,
}

#[cfg(feature = "std")]
impl BucketSlot {
    const fn new() -> Self {
        Self {
            buffers: [None, None, None, None, None, None, None, None],
            len: 0,
        }
    }

    #[inline]
    #[allow(
        unsafe_code,
        reason = "self.len is strictly bounded by MAX_PER_BUCKET: indexing is verified safe."
    )]
    fn push(&mut self, vec: Vec<Limb>) {
        if self.len < MAX_PER_BUCKET {
            // SAFETY: self.len < MAX_PER_BUCKET
            *unsafe { self.buffers.get_unchecked_mut(self.len) } = Some(vec);
            self.len = self.len.wrapping_add(1);
        }
    }

    #[inline]
    #[allow(
        clippy::indexing_slicing,
        reason = "self.len is strictly bounded by MAX_PER_BUCKET: indexing is verified safe and avoids branchy checks."
    )]
    const fn pop(&mut self) -> Option<Vec<Limb>> {
        if self.len > 0 {
            self.len = self.len.wrapping_sub(1);
            self.buffers[self.len].take()
        } else {
            None
        }
    }
}

#[cfg(all(feature = "std", arbi_eager_thread_local))]
thread_local! {
    /// Bucketed thread-local arena. Each bucket holds up to MAX_PER_BUCKET
    /// buffers, avoiding the O(n) linear scan in the hot path.
    /// The bucket array itself uses fixed-size structures to avoid internal heap allocations.
    static THREAD_SCRATCH_ARENA: RefCell<[BucketSlot; BUCKET_COUNT]> =
        const { RefCell::new([const { BucketSlot::new() }; BUCKET_COUNT]) };
}

// Rust cannot use eager const TLS when the target exposes only OS TLS keys.
// The necessarily lazy backend starts from the same empty fixed-size bucket.
#[cfg(all(feature = "std", not(arbi_eager_thread_local)))]
thread_local! {
    /// Bucketed thread-local arena for targets without native TLS storage.
    static THREAD_SCRATCH_ARENA: RefCell<[BucketSlot; BUCKET_COUNT]> =
        RefCell::from([const { BucketSlot::new() }; BUCKET_COUNT]);
}

/// A thread-local pool for scratch `Vec<Limb>` buffers.
/// This prevents repeated system allocations during algorithm loops.
///
/// For hot loops that repeatedly clone (e.g. BZ / Lehmer), prefer
/// [`ScratchBuffer::clone_into`] which reuses an existing buffer via
/// double-buffering instead of allocating a fresh one.
#[derive(Debug)]
pub struct ScratchBuffer {
    vec: Vec<Limb>,
}

impl Clone for ScratchBuffer {
    #[inline]
    fn clone(&self) -> Self {
        let mut new_buf = Self::acquire(self.capacity());
        new_buf.vec.extend_from_slice(&self.vec);
        new_buf
    }
}

impl ScratchBuffer {
    /// Acquires a buffer with at least `min_capacity` limbs from the
    /// thread-local arena.
    ///
    /// Bucketed lookup is constant-time and searches at most two larger
    /// buckets, preventing small requests from consuming massive buffers.
    #[must_use]
    #[inline]
    pub fn acquire(min_capacity: usize) -> Self {
        // Zero-capacity buffers are never pooled. Returning directly avoids a
        // TLS access for every initially empty algorithm scratch field.
        if min_capacity == 0 {
            return Self { vec: Vec::new() };
        }

        #[cfg(feature = "std")]
        {
            let start_bucket = bucket_for_capacity(min_capacity);
            let cached = THREAD_SCRATCH_ARENA.with(|arena| {
                let mut buckets = arena.borrow_mut();
                let end_bucket = min(start_bucket.wrapping_add(3), BUCKET_COUNT);
                #[allow(
                    unsafe_code,
                    reason = "bucket_idx is bounded by end_bucket <= BUCKET_COUNT"
                )]
                for bucket_idx in start_bucket..end_bucket {
                    // SAFETY: bucket_idx < BUCKET_COUNT
                    if let Some(vec) = unsafe { buckets.get_unchecked_mut(bucket_idx) }.pop() {
                        return Some(vec);
                    }
                }
                None
            });

            if let Some(mut vec) = cached {
                vec.clear();
                vec.reserve(min_capacity);
                return Self { vec };
            }
        }

        Self {
            vec: Vec::with_capacity(min_capacity),
        }
    }

    /// Discards the current contents and ensures capacity for at least
    /// `min_capacity` limbs, acquiring a size-matched pooled buffer when the
    /// current allocation is too small.
    ///
    /// Callers must no longer need the existing contents. This is preferable
    /// to `clear` followed by `resize` for one-shot arithmetic contexts: those
    /// contexts return large allocations to the arena on drop, and a later
    /// context must request the known capacity to retrieve them.
    pub fn reset_with_capacity(&mut self, min_capacity: usize) {
        if self.vec.capacity() < min_capacity {
            *self = Self::acquire(min_capacity);
        } else {
            self.vec.clear();
        }
    }

    /// Double-buffered clone: reuses `other`'s allocation for `self`, and
    /// copies `self`'s contents into `other`.  After the call, `other`
    /// holds a copy of `self` and `self` holds `other`'s old allocation
    /// (cleared and with capacity preserved).
    ///
    /// This avoids the allocation and TLS lookup inside [`Self::acquire`]
    /// on every iteration of hot loops like the BZ / Lehmer step.
    #[inline]
    pub fn clone_into(&mut self, other: &mut Self) {
        if other.capacity() < self.vec.len() {
            other
                .vec
                .reserve(self.vec.len().wrapping_sub(other.capacity()));
        }
        other.vec.clear();
        other.vec.extend_from_slice(&self.vec);
        swap(&mut self.vec, &mut other.vec);
    }

    /// Returns the current capacity of the internal buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.vec.capacity()
    }
}

impl Deref for ScratchBuffer {
    type Target = Vec<Limb>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl DerefMut for ScratchBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

#[cfg(feature = "std")]
impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        let mut vec = Vec::new();
        swap(&mut self.vec, &mut vec);

        // For small buffers, skip the TLS + RefCell overhead entirely.
        // Re-allocating a 64-limb buffer is cheaper than two atomic
        // operations (thread-local access + borrow).  The buffer is simply
        // dropped.
        if vec.capacity() < SMALL_BUFFER_DROP_THRESHOLD {
            return;
        }

        let bucket = bucket_for_capacity(vec.capacity());
        // Use `try_with` instead of `with`: on targets without eager TLS a
        // cached owner such as `FormatCache` registers its own TLS destructor,
        // and TLS destructors run in reverse order of first access. Because
        // the cache is first-accessed before the arena it fills, the cache is
        // destroyed *after* the arena, so returning a buffer from the cache's
        // destructor would hit the already-torn-down arena and panic with
        // `AccessError`, aborting thread teardown. When the arena is gone the
        // closure never runs, so `vec` is dropped with it and the pooled
        // capacity is released rather than returned to the pool.
        if THREAD_SCRATCH_ARENA
            .try_with(|arena| {
                let mut buckets = arena.borrow_mut();
                #[allow(
                    unsafe_code,
                    reason = "bucket is guaranteed to be < BUCKET_COUNT by bucket_for_capacity."
                )]
                // SAFETY: bucket < BUCKET_COUNT
                unsafe { buckets.get_unchecked_mut(bucket) }.push(vec);
            })
            .is_err()
        {
            // The arena TLS was already destroyed during thread teardown, so
            // the closure never ran and `vec` was dropped with it: the pooled
            // capacity is released rather than returned to a dead pool.
        }
    }
}
