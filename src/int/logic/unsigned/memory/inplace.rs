//! In-place memory management for the unsigned integer storage engine.

#![allow(
    unsafe_code,
    reason = "Bypassing bounds checks in custom memory layout updates to ensure completely branchless array manipulation and optimizer branch pruning."
)]

use core::{hint::unreachable_unchecked, mem::swap, ptr::copy_nonoverlapping};

use alloc::vec::Vec;

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};
impl InternalMpUint {
    /// In-place exchange of values without reallocation.
    /// Relies purely on native `core::mem::swap` for a zero-cost pointer swap.
    #[inline]
    pub const fn swap(&mut self, other: &mut Self) {
        swap(self, other);
    }

    /// Reserves capacity for at least `additional` more limbs.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match self.repr {
            UintRepr::Inline { ref len, ref limbs } => {
                let current_len = usize::from(*len);
                if current_len.saturating_add(additional) > INLINE_LIMBS {
                    let mut vec = Vec::with_capacity(current_len.saturating_add(additional));
                    // SAFETY: current_len <= INLINE_LIMBS, capacity is sufficient.
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
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        match self.repr {
            UintRepr::Inline { ref len, ref limbs } => {
                let current_len = usize::from(*len);
                if current_len.saturating_add(additional) > INLINE_LIMBS {
                    let mut vec = Vec::with_capacity(current_len.saturating_add(additional));
                    // SAFETY: current_len <= INLINE_LIMBS, capacity is sufficient.
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
    /// New limbs are filled with zeros.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        let current_len = self.limbs().len();
        if new_len <= current_len {
            match self.repr {
                UintRepr::Inline {
                    ref mut len,
                    ref mut limbs,
                } => {
                    // Zero out the trailing limbs beyond new_len in one shot.
                    // SAFETY: new_len <= current_len <= INLINE_LIMBS, so new_len..usize::from(*len)
                    // is within the limbs array bounds.
                    // SAFETY: new_len <= usize::from(*len) checked above, and
                    // usize::from(*len) <= INLINE_LIMBS by construction.
                    let tail = unsafe { limbs.get_unchecked_mut(new_len..usize::from(*len)) };
                    tail.fill(0);
                    // SAFETY: `new_len <= current_len <= INLINE_LIMBS`, and
                    // `INLINE_LIMBS = 4 <= u8::MAX` on every supported target.
                    *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                }
                UintRepr::Heap(ref mut vec) => vec.resize(new_len, 0),
            }
        } else {
            // Growing
            if new_len <= INLINE_LIMBS {
                match self.repr {
                    UintRepr::Inline {
                        ref mut len,
                        ref mut limbs,
                    } => {
                        // Zero-fill the new limbs in one shot.
                        // SAFETY: new_len <= INLINE_LIMBS and current_len <= new_len, so
                        // current_len..new_len is within the limbs array bounds.
                        let tail = unsafe { limbs.get_unchecked_mut(current_len..new_len) };
                        tail.fill(0);
                        // SAFETY: this branch proves
                        // `new_len <= INLINE_LIMBS = 4 <= u8::MAX`.
                        *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                    }
                    UintRepr::Heap(ref mut vec) => {
                        vec.resize(new_len, 0);
                        let mut limbs = [0; 4];
                        // SAFETY: new_len <= 4 (checked above) <= INLINE_LIMBS, so
                        // ..new_len is within the limbs array bounds.
                        let slice = unsafe { limbs.get_unchecked_mut(..new_len) };
                        // SAFETY: vec was resized to new_len.
                        slice.copy_from_slice(unsafe { vec.get_unchecked(..new_len) });
                        // SAFETY: this branch proves
                        // `new_len <= INLINE_LIMBS = 4 <= u8::MAX`.
                        let len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                        self.repr = UintRepr::Inline { len, limbs };
                    }
                }
            } else {
                // new_len > 4 -> MUST be Heap
                match self.repr {
                    UintRepr::Inline { ref len, ref limbs } => {
                        let mut vec = Vec::with_capacity(new_len);
                        // SAFETY: usize::from(*len) <= INLINE_LIMBS, so ..usize::from(*len)
                        // is within the limbs array bounds.
                        let slice = unsafe { limbs.get_unchecked(..usize::from(*len)) };
                        vec.extend_from_slice(slice);
                        vec.resize(new_len, 0);
                        self.repr = UintRepr::Heap(vec);
                    }
                    UintRepr::Heap(ref mut vec) => vec.resize(new_len, 0),
                }
            }
        }
    }

    /// Optimized hot-path: ensures `new_len` capacity, sets the length, and returns the mutable slice.
    /// Eliminates redundant branches on `UintRepr`.
    ///
    /// # Safety
    /// If `new_len` is greater than the previous length, the new elements are uninitialized memory.
    /// The caller must fully initialize them.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this allocation logic eliminates function call overhead and enables inter-procedural branch pruning."
    )]
    #[inline(always)]
    pub unsafe fn ensure_capacity_set_len_get_limbs(&mut self, new_len: usize) -> &mut [Limb] {
        if matches!(self.repr, UintRepr::Inline { .. }) {
            if new_len <= INLINE_LIMBS {
                if let UintRepr::Inline {
                    ref mut len,
                    ref mut limbs,
                } = self.repr
                {
                    // SAFETY: this branch proves
                    // `new_len <= INLINE_LIMBS = 4 <= u8::MAX`.
                    *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
                    // SAFETY: new_len <= INLINE_LIMBS
                    return unsafe { limbs.get_unchecked_mut(..new_len) };
                }
            } else {
                let mut vec = Vec::with_capacity(new_len);
                if let UintRepr::Inline { ref len, ref limbs } = self.repr {
                    let current_len = usize::from(*len);
                    // SAFETY: current_len <= INLINE_LIMBS < new_len
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), current_len);
                        vec.set_len(new_len);
                    }
                }
                self.repr = UintRepr::Heap(vec);
                if let UintRepr::Heap(ref mut v) = self.repr {
                    return v.as_mut_slice();
                }
            }
        } else if let UintRepr::Heap(ref mut vec) = self.repr {
            if vec.capacity() < new_len {
                vec.reserve(new_len.saturating_sub(vec.len()));
            }
            // SAFETY: vec has enough capacity
            unsafe {
                vec.set_len(new_len);
            }
            return vec.as_mut_slice();
        }
        // SAFETY: The repr is either Inline or Heap, which are exhaustively matched above.
        unsafe { unreachable_unchecked() }
    }

    /// Pushes a limb efficiently, switching from Inline to Heap if necessary.
    #[allow(
        clippy::inline_always,
        reason = "Inlining this allocation logic eliminates function call overhead and enables inter-procedural branch pruning."
    )]
    #[inline(always)]
    pub fn push_limb(&mut self, limb: Limb) {
        if matches!(self.repr, UintRepr::Inline { .. }) {
            let mut needs_heap = false;
            let mut current_len = 0;
            if let UintRepr::Inline { ref len, .. } = self.repr {
                current_len = usize::from(*len);
                needs_heap = current_len >= INLINE_LIMBS;
            }

            if needs_heap {
                let mut vec = Vec::with_capacity(INLINE_LIMBS.wrapping_add(1));
                if let UintRepr::Inline { ref limbs, .. } = self.repr {
                    // SAFETY: copying exactly INLINE_LIMBS
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), INLINE_LIMBS);
                        vec.set_len(INLINE_LIMBS);
                    }
                }
                vec.push(limb);
                self.repr = UintRepr::Heap(vec);
            } else if let UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } = self.repr
            {
                // SAFETY: current_len < INLINE_LIMBS
                unsafe {
                    *limbs.get_unchecked_mut(current_len) = limb;
                }
                *len = len.wrapping_add(1);
            }
        } else if let UintRepr::Heap(ref mut vec) = self.repr {
            vec.push(limb);
        }
    }
}

impl Clone for InternalMpUint {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
        }
    }

    /// Re-uses the existing vector allocation to avoid heap overhead.
    /// This acts as the Rust equivalent to C's `mpz_set`.
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (&mut self.repr, &source.repr) {
            // Fast path: both Inline — direct field copy, no allocation.
            (
                UintRepr::Inline {
                    len: dst_len,
                    limbs: dst_limbs,
                },
                UintRepr::Inline {
                    len: src_len,
                    limbs: src_limbs,
                },
            ) => {
                // SAFETY: both are Inline with INLINE_LIMBS elements; copying the
                // full array (including unused trailing limbs) is cheaper than a
                // conditional partial copy and still correct.
                *dst_len = *src_len;
                dst_limbs.copy_from_slice(src_limbs);
            }
            (UintRepr::Heap(dst), UintRepr::Heap(src)) => {
                dst.clone_from(src);
            }
            (UintRepr::Heap(dst), UintRepr::Inline { len, limbs }) => {
                dst.clear();
                // SAFETY: usize::from(*len) <= INLINE_LIMBS, so ..usize::from(*len)
                // is within the limbs array bounds.
                let slice = unsafe { limbs.get_unchecked(..usize::from(*len)) };
                dst.extend_from_slice(slice);
            }
            _ => {
                self.repr.clone_from(&source.repr);
            }
        }
    }
}
