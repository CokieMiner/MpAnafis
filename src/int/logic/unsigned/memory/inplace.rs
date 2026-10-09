//! In-place memory management for the unsigned integer storage engine.

#![expect(
    unsafe_code,
    reason = "Validated storage capacities and initialized prefixes bound raw copies, length commits, and inline access"
)]

use core::{
    mem::swap,
    ptr::{copy_nonoverlapping, write_bytes},
};

use alloc::vec::Vec;

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};

/// A destination span whose newly exposed heap limbs are not part of the value yet.
///
/// The owner keeps its original logical length until [`Self::commit`] is called.
/// This lets raw kernels initialize spare capacity without first constructing a
/// typed slice over uninitialized limbs.
#[derive(Debug)]
pub struct LimbWriteGuard<'storage> {
    owner: &'storage mut InternalMpUint,
    old_len: usize,
    new_len: usize,
}

impl<'storage> LimbWriteGuard<'storage> {
    /// Returns the start of the allocation for raw, write-only kernels.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut Limb {
        match self.owner.repr {
            UintRepr::Inline { ref mut limbs, .. } => limbs.as_mut_ptr(),
            UintRepr::Heap(ref mut limbs) => limbs.as_mut_ptr(),
        }
    }

    /// Initializes any newly exposed suffix with zeroes and commits the length.
    #[inline]
    #[must_use]
    pub fn initialize_suffix_with_zeroes(mut self) -> &'storage mut [Limb] {
        if self.new_len > self.old_len {
            // SAFETY: preparation reserves `new_len` limbs; old_len < new_len
            // proves the suffix length cannot underflow. The exclusive guard
            // owns this aligned allocation, and zero is a valid Limb value.
            unsafe {
                write_bytes(
                    self.as_mut_ptr().add(self.old_len),
                    0,
                    self.new_len.unchecked_sub(self.old_len),
                );
            }
        }
        // SAFETY: the old prefix was initialized by the representation invariant,
        // and the only newly exposed suffix was initialized immediately above.
        unsafe { self.commit() }
    }

    /// Commits the prepared logical length after a raw kernel initialized it.
    ///
    /// # Safety
    ///
    /// Every element in `0..new_len` must contain a valid initialized [`Limb`].
    /// Normalize the owner before magnitude queries if its highest limb is zero.
    #[inline]
    pub unsafe fn commit(self) -> &'storage mut [Limb] {
        let Self { owner, new_len, .. } = self;
        match owner.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                debug_assert!(
                    new_len <= INLINE_LIMBS,
                    "prepared inline length exceeds inline capacity"
                );
                // SAFETY: preparation proves the inline bound, and the caller
                // guarantees every exposed element is initialized.
                *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                // SAFETY: `new_len <= INLINE_LIMBS` by preparation.
                unsafe { limbs.get_unchecked_mut(..new_len) }
            }
            UintRepr::Heap(ref mut limbs) => {
                // SAFETY: preparation guaranteed capacity and the caller proved
                // initialization of the complete new logical span.
                unsafe {
                    limbs.set_len(new_len);
                }
                limbs.as_mut_slice()
            }
        }
    }
}

impl InternalMpUint {
    /// Exchanges representations without copying limbs or reallocating.
    #[inline]
    pub const fn swap(&mut self, other: &mut Self) {
        swap(self, other);
    }

    /// Reserves capacity for at least `additional` more limbs.
    ///
    /// # Panics
    /// Panics if the required limb count or allocation size is unrepresentable.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match self.repr {
            UintRepr::Inline { ref len, ref limbs } => {
                let current_len = usize::from(*len);
                let required = current_len
                    .checked_add(additional)
                    .expect("limb capacity overflow");
                if required > INLINE_LIMBS {
                    let mut vec = Vec::with_capacity(required);
                    // SAFETY: current_len <= INLINE_LIMBS bounds the initialized
                    // source; the fresh exclusive allocation reserves required
                    // >= current_len disjoint slots. Copying initializes the committed prefix.
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), current_len);
                        vec.set_len(current_len);
                    }
                    self.repr = UintRepr::Heap(vec);
                }
            }
            UintRepr::Heap(ref mut vec) => vec.reserve(additional),
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more limbs.
    ///
    /// # Panics
    /// Panics if the required limb count or allocation size is unrepresentable.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        match self.repr {
            UintRepr::Inline { ref len, ref limbs } => {
                let current_len = usize::from(*len);
                let required = current_len
                    .checked_add(additional)
                    .expect("limb capacity overflow");
                if required > INLINE_LIMBS {
                    let mut vec = Vec::with_capacity(required);
                    // SAFETY: current_len <= INLINE_LIMBS bounds the initialized
                    // source; the fresh exclusive allocation reserves required
                    // >= current_len disjoint slots. Copying initializes the committed prefix.
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), current_len);
                        vec.set_len(current_len);
                    }
                    self.repr = UintRepr::Heap(vec);
                }
            }
            UintRepr::Heap(ref mut vec) => vec.reserve_exact(additional),
        }
    }

    /// Returns the number of limbs the vector can hold without reallocating.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        match self.repr {
            UintRepr::Inline { .. } => INLINE_LIMBS,
            UintRepr::Heap(ref vec) => vec.capacity(),
        }
    }

    /// Shrinks the capacity of the integer as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if let UintRepr::Heap(ref mut vec) = self.repr {
            vec.shrink_to_fit();
        }
    }

    /// Resizes the internal representation to exactly `new_len` limbs.
    /// New limbs are filled with zeros. Normalize before magnitude queries.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                let current_len = usize::from(*len);
                if new_len <= INLINE_LIMBS {
                    // Inactive inline slots are outside the value; growth
                    // initializes every newly exposed slot independently.
                    if new_len > current_len {
                        // Zero-fill the new limbs in one shot.
                        // SAFETY: new_len <= INLINE_LIMBS and current_len < new_len, so
                        // current_len..new_len is within the limbs array bounds.
                        let tail = unsafe { limbs.get_unchecked_mut(current_len..new_len) };
                        tail.fill(0);
                    }
                    // SAFETY: `new_len <= INLINE_LIMBS = 4 <= u8::MAX`.
                    *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                } else {
                    // Transition Inline -> Heap
                    let mut vec = Vec::with_capacity(new_len);
                    // SAFETY: current_len <= INLINE_LIMBS bounds the initialized prefix.
                    let slice = unsafe { limbs.get_unchecked(..current_len) };
                    vec.extend_from_slice(slice);
                    vec.resize(new_len, 0);
                    self.repr = UintRepr::Heap(vec);
                }
            }
            UintRepr::Heap(ref mut vec) => {
                vec.resize(new_len, 0);
            }
        }
    }

    /// Prepares an already allocated heap buffer for complete overwrite.
    ///
    /// Returns `None` when the value is inline or the allocation is too small,
    /// leaving `self` unchanged so the caller can use the general growth path.
    ///
    #[expect(
        clippy::inline_always,
        reason = "The reusable-heap probe removes repeated representation dispatch from destination-reusing arithmetic"
    )]
    #[inline(always)]
    pub const fn try_prepare_reused_heap_limbs(
        &mut self,
        new_len: usize,
    ) -> Option<LimbWriteGuard<'_>> {
        let UintRepr::Heap(ref limbs) = self.repr else {
            return None;
        };
        if limbs.capacity() < new_len {
            return None;
        }
        let old_len = limbs.len();
        Some(LimbWriteGuard {
            owner: self,
            old_len,
            new_len,
        })
    }

    /// Ensures capacity and prepares `new_len` limbs without exposing an
    /// uninitialized typed slice.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this allocation logic eliminates function call overhead and enables inter-procedural branch pruning."
    )]
    #[inline(always)]
    pub fn prepare_limb_write(&mut self, new_len: usize) -> LimbWriteGuard<'_> {
        let old_len = self.limbs().len();
        match self.repr {
            UintRepr::Inline { ref len, ref limbs } if new_len > INLINE_LIMBS => {
                let mut vec = Vec::with_capacity(new_len);
                let current_len = usize::from(*len);
                // SAFETY: the source has `current_len <= INLINE_LIMBS`
                // initialized limbs and the new allocation has enough capacity.
                unsafe {
                    copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), current_len);
                    vec.set_len(current_len);
                }
                self.repr = UintRepr::Heap(vec);
            }
            UintRepr::Heap(ref mut limbs) if limbs.capacity() < new_len => {
                // SAFETY: len <= capacity < new_len, so the required additional
                // count is positive and representable on every pointer width.
                let additional = unsafe { new_len.unchecked_sub(limbs.len()) };
                limbs.reserve(additional);
            }
            UintRepr::Inline { .. } | UintRepr::Heap(_) => {}
        }
        LimbWriteGuard {
            owner: self,
            old_len,
            new_len,
        }
    }

    /// Ensures `new_len` initialized writable limbs, preserving the old prefix.
    #[inline]
    pub fn ensure_capacity_set_len_get_limbs(&mut self, new_len: usize) -> &mut [Limb] {
        self.prepare_limb_write(new_len)
            .initialize_suffix_with_zeroes()
    }

    /// Appends a high limb, growing beyond the four-limb inline capacity.
    ///
    /// Appending zero to a nonzero value requires normalization before queries.
    #[expect(
        clippy::inline_always,
        reason = "Inlining this allocation logic eliminates function call overhead and enables inter-procedural branch pruning."
    )]
    #[inline(always)]
    pub fn push_limb(&mut self, limb: Limb) {
        if limb == 0 && self.is_zero() {
            return;
        }

        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                let current_len = usize::from(*len);
                if current_len < INLINE_LIMBS {
                    // SAFETY: current_len < INLINE_LIMBS == limbs.len()
                    unsafe {
                        *limbs.get_unchecked_mut(current_len) = limb;
                    }
                    // SAFETY: len < INLINE_LIMBS = 4, so len + 1 <= 4 fits u8.
                    *len = unsafe { len.unchecked_add(1) };
                } else {
                    let mut vec = Vec::with_capacity(INLINE_LIMBS + 1);
                    // SAFETY: current_len == INLINE_LIMBS; copying full inline capacity
                    // and writing the new limb into the reserved slot at index INLINE_LIMBS.
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), INLINE_LIMBS);
                        *vec.as_mut_ptr().add(INLINE_LIMBS) = limb;
                        vec.set_len(INLINE_LIMBS + 1);
                    }
                    self.repr = UintRepr::Heap(vec);
                }
            }
            UintRepr::Heap(ref mut vec) => {
                vec.push(limb);
            }
        }
    }
}

impl Clone for InternalMpUint {
    #[inline]
    fn clone(&self) -> Self {
        if let UintRepr::Heap(limbs) = &self.repr
            && limbs.len() <= INLINE_LIMBS
        {
            let mut inline = [0; INLINE_LIMBS];
            // SAFETY: the heap's initialized length is at most INLINE_LIMBS;
            // the fresh inline array is disjoint and has room for that prefix.
            unsafe {
                copy_nonoverlapping(limbs.as_ptr(), inline.as_mut_ptr(), limbs.len());
            }
            // SAFETY: limbs.len() <= INLINE_LIMBS == 4 fits in u8 on all targets.
            let len = unsafe { u8::try_from(limbs.len()).unwrap_unchecked() };
            return Self {
                repr: UintRepr::Inline { len, limbs: inline },
            };
        }
        Self {
            repr: self.repr.clone(),
        }
    }

    /// Copies the magnitude while retaining an existing heap allocation.
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        if matches!(self.repr, UintRepr::Heap(_)) {
            self.repr.clone_from(&source.repr);
        } else {
            // An inline destination has no allocation to preserve; clone()
            // also maps short heap sources directly to inline storage.
            *self = source.clone();
        }
    }
}
