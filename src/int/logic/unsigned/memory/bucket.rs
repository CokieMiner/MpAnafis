//! Fixed-capacity occupied-prefix storage for the thread-local scratch arena.

#![expect(
    unsafe_code,
    reason = "The occupied prefix bounds initialized Some entries; extraction and last-slot compaction preserve the empty suffix"
)]

use alloc::vec::Vec;

use super::Limb;

/// Maximum occupancy of one arena bucket.
pub const MAX_PER_BUCKET: usize = 16;

/// `len <= MAX_PER_BUCKET`; the prefix contains `Some` and the suffix `None`.
#[derive(Debug)]
pub struct BucketSlot {
    buffers: [Option<Vec<Limb>>; MAX_PER_BUCKET],
    len: usize,
}

impl BucketSlot {
    /// Constructs an empty occupied prefix.
    pub const fn new() -> Self {
        Self {
            buffers: [const { None }; MAX_PER_BUCKET],
            len: 0,
        }
    }

    #[inline]
    /// Retains `vec` when occupancy is below `retention_limit <= MAX_PER_BUCKET`.
    pub fn push(&mut self, vec: Vec<Limb>, retention_limit: usize) {
        debug_assert!(
            self.len <= MAX_PER_BUCKET && retention_limit <= MAX_PER_BUCKET,
            "bucket occupancy and retention must fit the fixed storage"
        );
        if self.len < retention_limit {
            // SAFETY: max_buffers_for_bucket supplies retention_limit <=
            // MAX_PER_BUCKET. The old suffix is None, so this first suffix slot
            // owns no allocation to drop. A raw write avoids testing that proved
            // empty state before replacing it with the initialized Some entry.
            unsafe {
                self.buffers.as_mut_ptr().add(self.len).write(Some(vec));
            }
            // SAFETY: len < retention_limit <= MAX_PER_BUCKET = 16, so len + 1 fits.
            self.len = unsafe { self.len.unchecked_add(1) };
        }
    }

    #[inline]
    /// Removes the last occupied slot, or returns `None` for an empty bucket.
    pub fn pop(&mut self) -> Option<Vec<Limb>> {
        debug_assert!(
            self.len <= MAX_PER_BUCKET,
            "bucket occupancy fits its storage"
        );
        if self.len == 0 {
            return None;
        }
        // SAFETY: len > 0 and len <= MAX_PER_BUCKET. Decrementing makes
        // the old last occupied index strictly less than MAX_PER_BUCKET.
        self.len = unsafe { self.len.unchecked_sub(1) };
        // SAFETY: the preceding decrement selects the old last occupied
        // slot, below MAX_PER_BUCKET. The occupied-prefix invariant proves
        // Some; taking it restores None in the suffix. Unchecked extraction
        // keeps a successful pop from propagating an impossible empty slot.
        Some(unsafe {
            self.buffers
                .get_unchecked_mut(self.len)
                .take()
                .unwrap_unchecked()
        })
    }

    /// Extracts the smallest buffer satisfying `capacity >= min_capacity`.
    ///
    /// Reverse scanning chooses the highest index on ties. Removing that slot
    /// needs no compaction; an interior match receives the old last entry.
    #[inline]
    pub fn take_best_fit(&mut self, min_capacity: usize) -> Option<Vec<Limb>> {
        debug_assert!(
            self.len <= MAX_PER_BUCKET,
            "bucket occupancy fits its storage"
        );
        let mut best_idx = None;
        let mut best_cap = usize::MAX;

        for i in (0..self.len).rev() {
            // SAFETY: i < len <= MAX_PER_BUCKET bounds the array access;
            // the occupied-prefix invariant proves this entry is Some.
            // The immutable borrow cannot change occupancy during the scan.
            let buf = unsafe { self.buffers.get_unchecked(i).as_ref().unwrap_unchecked() };
            let cap = buf.capacity();
            if cap >= min_capacity && cap < best_cap {
                best_cap = cap;
                best_idx = Some(i);
                if cap == min_capacity {
                    break;
                }
            }
        }

        let idx = best_idx?;
        // SAFETY: finding an index proves the old len is positive. Removing
        // one occupied entry gives the last index used for suffix compaction.
        self.len = unsafe { self.len.unchecked_sub(1) };
        // SAFETY: the scan selected idx in the old occupied prefix. Changing
        // len does not alter its initialized Some slot, and idx <= new len <
        // MAX_PER_BUCKET. Extraction cannot fail on a matching path.
        let found = unsafe {
            self.buffers
                .get_unchecked_mut(idx)
                .take()
                .unwrap_unchecked()
        };
        if idx != self.len {
            // SAFETY: idx < new len selects a different slot from the old last
            // occupied entry. That entry is still Some and below MAX_PER_BUCKET;
            // taking it clears the new suffix without an additional Option check.
            let last = unsafe {
                self.buffers
                    .get_unchecked_mut(self.len)
                    .take()
                    .unwrap_unchecked()
            };
            // SAFETY: idx < MAX_PER_BUCKET and the earlier take left None there.
            // Its old value owns no allocation, so writing needs no drop check.
            unsafe {
                self.buffers.as_mut_ptr().add(idx).write(Some(last));
            }
        }
        Some(found)
    }
}
