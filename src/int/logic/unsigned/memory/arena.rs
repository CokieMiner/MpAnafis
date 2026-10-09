//! Thread-local scratch-buffer arena for reusable unsigned arithmetic storage.
//!
//! Reusable copies preserve destination allocations. Only insufficient capacity
//! enters pooled replacement, which acquires storage, copies initialized limbs,
//! and releases the old allocation through the arena's retention policy.
//! Retention limits govern released allocations; live buffers retain their capacity.

#![expect(
    unsafe_code,
    reason = "arena occupancy bounds initialized bucket access, and acquisition guarantees capacity for disjoint initialized limb copies"
)]

#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use core::cmp::min;
#[cfg(feature = "std")]
use core::mem::take;
#[cfg(feature = "std")]
use core::num::NonZeroUsize;
use core::{
    mem::swap,
    ops::{Deref, DerefMut},
    ptr::copy_nonoverlapping,
};
#[cfg(feature = "std")]
use std::thread_local;

use alloc::vec::Vec;

use super::Limb;
#[cfg(feature = "std")]
use super::{BucketSlot, MAX_PER_BUCKET};

/// Bucket count, with an exclusive power-of-two limit representable in usize.
/// On 16-bit pointers, `isize::MAX / size_of::<Limb>() = 2^14 - 1` limbs;
/// the exclusive limit is 2^14. Wider pointers retain the 2^21 pool limit.
#[cfg(feature = "std")]
const BUCKET_COUNT: usize = if usize::BITS == 16 { 14 } else { 21 };

/// Capacities below 64 limbs are not retained by the arena.
#[cfg(feature = "std")]
const SMALL_BUFFER_DROP_THRESHOLD: usize = 64;
/// Number of larger power-of-two buckets eligible for one request.
#[cfg(feature = "std")]
const POOL_LOOKAHEAD_BUCKETS: usize = 2;

/// Exclusive capacity limit for retained buffers: `2^14` on 16-bit pointers
/// and `2^21` on 32- and 64-bit pointers.
#[cfg(feature = "std")]
const MAX_POOLED_CAPACITY: usize = 1 << BUCKET_COUNT;

#[cfg(all(feature = "std", mp_eager_thread_local))]
thread_local! {
    static THREAD_SCRATCH_ARENA: RefCell<[BucketSlot; BUCKET_COUNT]> =
        const { RefCell::new([const { BucketSlot::new() }; BUCKET_COUNT]) };
}

// Targets with OS-managed TLS keys require lazy arena initialization.
#[cfg(all(feature = "std", not(mp_eager_thread_local)))]
thread_local! {
    static THREAD_SCRATCH_ARENA: RefCell<[BucketSlot; BUCKET_COUNT]> =
        RefCell::from([const { BucketSlot::new() }; BUCKET_COUNT]);
}

/// An owned limb buffer acquired from the thread-local arena when available.
///
/// Repeated copies can reuse capacity through [`Self::clone_from`];
/// [`Self::clone_into`] also exchanges the two buffers after copying.
#[derive(Debug)]
pub struct ScratchBuffer {
    vec: Vec<Limb>,
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
        Self {
            vec: Self::acquire_vec(min_capacity),
        }
    }

    /// Acquires a pooled limb allocation with at least `min_capacity` limbs.
    ///
    /// This is the shared primitive behind [`Self::acquire`] and reusable
    /// algorithm scratch buffers: algorithm scratch spaces recycle thread-local
    /// capacity instead of returning to the system allocator on every call.
    /// Only capacity is pooled; the returned vector is always empty, so callers
    /// must initialize before reading.
    #[must_use]
    #[inline]
    fn acquire_vec(min_capacity: usize) -> Vec<Limb> {
        #[cfg(feature = "std")]
        {
            // Below 64/4, the initial bucket and its two successors all hold
            // capacities below 64, which release_vec never retains. Requests
            // at or above the exclusive pool limit cannot reuse pooled capacity.
            if !(SMALL_BUFFER_DROP_THRESHOLD >> POOL_LOOKAHEAD_BUCKETS..MAX_POOLED_CAPACITY)
                .contains(&min_capacity)
            {
                return Vec::with_capacity(min_capacity);
            }
            Self::acquire_vec_slow(min_capacity)
        }

        #[cfg(not(feature = "std"))]
        {
            Vec::with_capacity(min_capacity)
        }
    }

    /// Returns a limb allocation to the thread-local pool for reuse.
    ///
    /// Small buffers drop immediately; larger ones are retained subject to the
    /// per-bucket bound. This is the shared primitive behind `Drop for ScratchBuffer`,
    /// keeping one retention policy for every pooled scratch allocation.
    #[inline]
    #[cfg(feature = "std")]
    pub fn release_vec(vec: Vec<Limb>) {
        if !(SMALL_BUFFER_DROP_THRESHOLD..MAX_POOLED_CAPACITY).contains(&vec.capacity()) {
            return;
        }
        Self::release_vec_slow(vec);
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

    /// Double-buffered clone: copies `self` into `other`'s allocation, then
    /// swaps the allocations. After the call both buffers contain the original
    /// value of `self`; `self` owns `other`'s former allocation and `other`
    /// owns `self`'s former allocation.
    ///
    /// Adequate destination capacity avoids allocation and arena lookup.
    /// Otherwise, `other` grows through `Vec` according to the source length.
    #[inline]
    pub fn clone_into(&mut self, other: &mut Self) {
        // extend_from_slice reserves against the cleared length. Reserving
        // against the old length first can grow twice or overallocate.
        other.vec.clear();
        other.vec.extend_from_slice(&self.vec);
        swap(&mut self.vec, &mut other.vec);
    }

    /// Returns the current capacity of the internal buffer.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    // --- Specialized Internal Routines ---

    /// Slow-path limb acquisition from the thread-local bucketed arena.
    ///
    /// Keeps TLS access, borrow checks, and bucket scans out of allocation sites.
    #[cfg(feature = "std")]
    #[inline(never)]
    fn acquire_vec_slow(min_capacity: usize) -> Vec<Limb> {
        let start_bucket = bucket_for_pooled_capacity(min_capacity);
        let cached = THREAD_SCRATCH_ARENA
            .try_with(|arena| {
                let mut buckets = arena.try_borrow_mut().ok()?;
                // SAFETY: bucket_for_pooled_capacity yields start_bucket <
                // BUCKET_COUNT <= 21, so adding three fits every pointer width.
                let search_end = unsafe { start_bucket.unchecked_add(POOL_LOOKAHEAD_BUCKETS + 1) };
                let end_bucket = min(search_end, BUCKET_COUNT);

                // 1. In the initial bucket, search for the best fit (smallest capacity >= min_capacity)
                // to avoid reallocating when an adequate buffer is already pooled.
                // SAFETY: the validated request yields start_bucket < BUCKET_COUNT.
                let start_slot = unsafe { buckets.get_unchecked_mut(start_bucket) };
                if let Some(vec) = start_slot.take_best_fit(min_capacity) {
                    return Some(vec);
                }

                // 2. In higher buckets, power-of-two geometry guarantees capacity > min_capacity,
                // so a simple pop() is always sufficient without reallocation.
                // SAFETY: start_bucket < BUCKET_COUNT <= 21, so its successor fits.
                let higher_start = unsafe { start_bucket.unchecked_add(1) };
                for bucket_idx in higher_start..end_bucket {
                    // SAFETY: bucket_idx < BUCKET_COUNT
                    if let Some(vec) = unsafe { buckets.get_unchecked_mut(bucket_idx) }.pop() {
                        return Some(vec);
                    }
                }

                // 3. Fallback: take any available buffer from start_bucket to grow via reserve.
                // SAFETY: start_bucket < BUCKET_COUNT
                unsafe { buckets.get_unchecked_mut(start_bucket) }.pop()
            })
            .ok()
            .flatten();

        if let Some(mut vec) = cached {
            vec.clear();
            if vec.capacity() < min_capacity {
                if start_bucket == BUCKET_COUNT - 1 {
                    vec.reserve_exact(min_capacity);
                } else {
                    vec.reserve(min_capacity);
                }
            }
            return vec;
        }

        Vec::with_capacity(min_capacity)
    }

    /// Slow-path limb buffer retention in the thread-local bucketed arena.
    ///
    /// Keeps TLS access and borrow checks out of arithmetic unwind cleanup.
    #[cfg(feature = "std")]
    #[inline(never)]
    fn release_vec_slow(vec: Vec<Limb>) {
        let bucket = bucket_for_pooled_capacity(vec.capacity());
        let retention_limit = max_buffers_for_bucket(bucket);

        // TLS owners can release buffers after the arena's destructor, and
        // releases can occur while another arena borrow is active. Both are
        // valid deallocation paths: try_with and try_borrow_mut let the owned
        // vector drop without pooling or a panic during unwinding.
        #[expect(
            clippy::let_underscore_must_use,
            reason = "Thread-local arena teardown or contention safely drops the allocation without pooling."
        )]
        let _ = THREAD_SCRATCH_ARENA.try_with(|arena| {
            let Ok(mut buckets) = arena.try_borrow_mut() else {
                return;
            };
            // SAFETY: bucket < BUCKET_COUNT
            unsafe { buckets.get_unchecked_mut(bucket) }.push(vec, retention_limit);
        });
    }
}

// ============================================================================
// Standard Trait Implementations
// ============================================================================

impl Clone for ScratchBuffer {
    /// Clones initialized limbs into independently owned storage with at least
    /// the source capacity, acquiring pooled storage when available.
    #[inline]
    fn clone(&self) -> Self {
        let mut new_buf = Self::acquire(self.capacity());
        let len = self.vec.len();
        // SAFETY: acquire returns an empty, exclusively owned vector with
        // capacity >= self.capacity() >= len. Its allocation is disjoint
        // from the live source, whose entire len-limb prefix is initialized.
        // Both pointers are aligned and nonnull even for len == 0.
        // Limb is usize, so bitwise duplication preserves independent Copy values.
        // Copying initializes exactly the prefix exposed by
        // set_len, eliminating an already-proved reserve check.
        unsafe {
            copy_nonoverlapping(self.vec.as_ptr(), new_buf.vec.as_mut_ptr(), len);
            new_buf.vec.set_len(len);
        }
        new_buf
    }

    /// Copies the initialized contents, reusing destination capacity or
    /// acquiring a pooled replacement when the destination is too small.
    /// Arena acquisition and release occur only on the replacement path.
    /// Unlike `clone`, this need not reproduce the source's spare capacity.
    /// Sufficient destination capacity is retained even for an empty source.
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        let len = source.vec.len();
        if self.vec.capacity() < len {
            replace_with_pooled_clone(self, source);
        } else {
            // SAFETY: the capacity test reserves the complete len-limb output.
            // The shared source is initialized and cannot overlap the exclusively
            // borrowed destination. Both pointers are aligned and nonnull, even
            // for len == 0. Copying initializes the entire committed prefix;
            // shortening discards only Copy limbs, which have no destructor.
            unsafe {
                copy_nonoverlapping(source.vec.as_ptr(), self.vec.as_mut_ptr(), len);
                self.vec.set_len(len);
            }
        }
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
        let vec = take(&mut self.vec);
        Self::release_vec(vec);
    }
}

/// Replaces insufficient destination capacity through the arena.
///
/// Outlining keeps allocation and unwind cleanup outside reusable copies.
/// Assignment invokes `clone`; `clone_from` would redispatch to this same path.
#[inline(never)]
#[expect(
    clippy::assigning_clones,
    reason = "clone_from dispatches insufficient capacity here; direct assignment acquires pooled storage without recursive dispatch"
)]
fn replace_with_pooled_clone(destination: &mut ScratchBuffer, source: &ScratchBuffer) {
    *destination = source.clone();
}

/// Returns floor(log2(cap)) for a validated nonzero capacity below the pool limit.
#[cfg(feature = "std")]
#[inline]
#[expect(
    clippy::as_conversions,
    reason = "Validated pool capacities have logarithms below BUCKET_COUNT <= 21, fitting usize on 16-, 32-, and 64-bit pointers"
)]
const fn bucket_for_pooled_capacity(cap: usize) -> usize {
    // SAFETY: acquire_vec validates cap >= 16; release_vec validates cap >= 64.
    // Both reject cap >= 2^BUCKET_COUNT before calling their slow paths, the
    // only callers here. Thus cap != 0 and log2(cap) < BUCKET_COUNT.
    unsafe { NonZeroUsize::new_unchecked(cap) }.ilog2() as usize
}

/// Retains at most 16, four, or two buffers as the capacity bucket increases.
#[cfg(feature = "std")]
#[inline]
const fn max_buffers_for_bucket(bucket: usize) -> usize {
    match bucket {
        0..=15 => MAX_PER_BUCKET,
        16..=18 => MAX_PER_BUCKET.div_euclid(4),
        // bucket_for_pooled_capacity bounds the remaining indices by 20.
        _ => MAX_PER_BUCKET.div_euclid(8),
    }
}
