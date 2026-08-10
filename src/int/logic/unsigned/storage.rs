//! Unsigned integer storage representation and canonical limb management.

#![allow(
    unsafe_code,
    reason = "Limb ops require raw pointers for peak assembly performance"
)]
use core::ptr::copy_nonoverlapping;

use alloc::vec::Vec;

use super::{INLINE_LIMBS, Limb};

/// The internal representation of the magnitude.
#[derive(Debug, PartialEq, Eq, Hash)]
#[doc(hidden)]
#[non_exhaustive]
pub enum UintRepr {
    /// Small-Inline representation with `INLINE_LIMBS` limbs on the stack.
    Inline {
        /// Number of valid limbs (0 to `INLINE_LIMBS`).
        len: u8,
        /// The inline limbs.
        limbs: [Limb; INLINE_LIMBS],
    },
    /// Heap-allocated representation for arbitrary precision.
    Heap(Vec<Limb>),
}

impl Clone for UintRepr {
    fn clone(&self) -> Self {
        match self {
            Self::Inline { len, limbs } => Self::Inline {
                len: *len,
                limbs: *limbs,
            },
            Self::Heap(vec) => Self::Heap(vec.clone()),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Self::Heap(dest_vec), Self::Heap(src_vec)) => {
                dest_vec.clone_from(src_vec); // Reuses the heap allocation!
            }
            (dest, src) => {
                *dest = src.clone();
            }
        }
    }
}

/// The core unsigned arbitrary precision integer engine.
#[derive(Debug)]
pub struct InternalArbiUint {
    pub repr: UintRepr,
}

impl InternalArbiUint {
    /// Creates a new unsigned integer with the value 0.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            repr: UintRepr::Inline {
                len: 0,
                limbs: [0; INLINE_LIMBS],
            },
        }
    }

    /// Creates a new unsigned integer with the value 1.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self {
            repr: UintRepr::Inline {
                len: 1,
                limbs: [1, 0, 0, 0],
            },
        }
    }

    /// Pre-allocates memory for a specific number of limbs.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= INLINE_LIMBS {
            Self::zero()
        } else {
            Self {
                repr: UintRepr::Heap(Vec::with_capacity(capacity)),
            }
        }
    }

    /// Same as `from_limbs` but skips the trailing-zero scan.
    ///
    /// # Safety
    /// The caller must ensure the highest limb in `limbs` is non-zero (or
    /// `limbs` is empty, which produces zero).
    #[allow(clippy::inline_always, reason = "hot path constructor")]
    #[inline(always)]
    #[must_use]
    pub unsafe fn from_limbs_normalized(limbs: Vec<Limb>) -> Self {
        if limbs.is_empty() {
            return Self::zero();
        }
        if limbs.len() <= INLINE_LIMBS {
            let mut arr = [0; INLINE_LIMBS];
            let copy_len = limbs.len();
            // SAFETY: copy_len <= INLINE_LIMBS <= 4, valid for copy_len and fits in u8 without overflow.
            let len = unsafe {
                copy_nonoverlapping(limbs.as_ptr(), arr.as_mut_ptr(), copy_len);
                u8::try_from(copy_len).unwrap_unchecked()
            };
            Self {
                repr: UintRepr::Inline { len, limbs: arr },
            }
        } else {
            Self {
                repr: UintRepr::Heap(limbs),
            }
        }
    }

    /// Constructs an integer from a vector of limbs, stripping trailing zeros and optimizing representation.
    #[allow(
        clippy::inline_always,
        reason = "from_limbs is a core constructor called on every operation result"
    )]
    #[inline(always)]
    #[must_use]
    pub fn from_limbs(mut limbs: Vec<Limb>) -> Self {
        // Find the last non-zero limb position and truncate in one operation.
        if let Some(last_nonzero) = limbs.iter().rposition(|&l| l != 0) {
            limbs.truncate(last_nonzero.wrapping_add(1));
        } else {
            return Self::zero();
        }
        if limbs.len() <= INLINE_LIMBS {
            let mut arr = [0; INLINE_LIMBS];
            let copy_len = limbs.len();
            // SAFETY: copy_len <= INLINE_LIMBS <= limbs.len() (guarded above and by truncate).
            // Both pointers are valid for `copy_len` reads/writes of properly aligned Limb values.
            // copy_len <= 4 always fits in u8 without overflow.
            let len = unsafe {
                copy_nonoverlapping(limbs.as_ptr(), arr.as_mut_ptr(), copy_len);
                u8::try_from(copy_len).unwrap_unchecked()
            };
            return Self {
                repr: UintRepr::Inline { len, limbs: arr },
            };
        }
        Self {
            repr: UintRepr::Heap(limbs),
        }
    }

    /// Replaces this value with the normalized limbs from `slice`.
    pub fn clone_from_slice(&mut self, slice: &[Limb]) {
        let mut len = slice.len();
        // SAFETY: The bounds check is explicit: len > 0 ensures len.wrapping_sub(1) is valid in the slice.
        while len > 0 && unsafe { *slice.get_unchecked(len.wrapping_sub(1)) } == 0 {
            len = len.wrapping_sub(1);
        }
        if len == 0 {
            self.clear();
            return;
        }
        if len <= INLINE_LIMBS {
            let mut arr = [0; INLINE_LIMBS];
            // SAFETY: We verified len <= INLINE_LIMBS and len <= slice.len()
            unsafe {
                copy_nonoverlapping(slice.as_ptr(), arr.as_mut_ptr(), len);
            }
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "inline limb count is at most INLINE_LIMBS — always fits in u8"
            )]
            let len_u8 = len as u8;
            self.repr = UintRepr::Inline {
                len: len_u8,
                limbs: arr,
            };
            return;
        }
        // SAFETY: len <= slice.len() is guaranteed by the trimming loop above
        let trimmed_slice = unsafe { slice.get_unchecked(..len) };
        match self.repr {
            UintRepr::Heap(ref mut vec) => {
                vec.clear();
                vec.extend_from_slice(trimmed_slice);
            }
            UintRepr::Inline { .. } => {
                self.repr = UintRepr::Heap(trimmed_slice.to_vec());
            }
        }
    }

    /// Sets the length of the internal representation without initializing the memory.
    /// # Safety
    /// The caller must ensure that the capacity is sufficient and that the elements
    /// up to `new_len` are properly initialized before being read.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        match self.repr {
            UintRepr::Inline { ref mut len, .. } => {
                debug_assert!(
                    new_len <= INLINE_LIMBS,
                    "inline length must not exceed INLINE_LIMBS"
                );
                // SAFETY: the unsafe caller guarantees `new_len` does not
                // exceed this representation's capacity. In the inline arm
                // that capacity is `INLINE_LIMBS = 4 <= u8::MAX`.
                *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
            }
            UintRepr::Heap(ref mut vec) => {
                // SAFETY: Caller guarantees capacity and initialization
                unsafe {
                    vec.set_len(new_len);
                }
            }
        }
    }

    /// Strips leading zero limbs to ensure canonical representation.
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "normalize is called in every arithmetic hot path"
    )]
    pub fn normalize(&mut self) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                while *len > 0 {
                    // SAFETY: The loop condition verifies *len > 0 before computing *len - 1,
                    // so the index is always in bounds 0..INLINE_LIMBS.
                    let is_zero =
                        unsafe { *limbs.get_unchecked(usize::from(len.wrapping_sub(1))) } == 0;
                    if !is_zero {
                        break;
                    }
                    *len = len.wrapping_sub(1);
                }
            }
            UintRepr::Heap(ref mut vec) => {
                if let Some(last_nz) = vec.iter().rposition(|&l| l != 0) {
                    vec.truncate(last_nz.wrapping_add(1));
                } else {
                    vec.clear();
                }
            }
        }
    }

    /// Clears the value to zero, retaining allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        match self.repr {
            UintRepr::Inline { ref mut len, .. } => {
                *len = 0;
            }
            UintRepr::Heap(ref mut vec) => {
                vec.clear();
            }
        }
    }

    /// Adds 1 in-place.
    #[inline]
    pub fn increment(&mut self) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                let old_len = usize::from(*len);
                // SAFETY: old_len <= INLINE_LIMBS, verified at construction
                let slice = unsafe { limbs.get_unchecked_mut(..old_len) };
                for limb in slice {
                    let (sum, overflow) = limb.overflowing_add(1);
                    *limb = sum;
                    if !overflow {
                        return;
                    }
                }
                if old_len < INLINE_LIMBS {
                    // SAFETY: old_len < INLINE_LIMBS, index is in bounds
                    unsafe {
                        *limbs.get_unchecked_mut(old_len) = 1;
                    }
                    *len = len.wrapping_add(1);
                } else {
                    let mut vec = Vec::with_capacity(INLINE_LIMBS.wrapping_add(1));
                    // SAFETY: copying exactly INLINE_LIMBS limbs
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), INLINE_LIMBS);
                        vec.set_len(INLINE_LIMBS);
                    }
                    vec.push(1);
                    self.repr = UintRepr::Heap(vec);
                }
            }
            UintRepr::Heap(ref mut vec) => {
                for limb in vec.iter_mut() {
                    let (sum, overflow) = limb.overflowing_add(1);
                    *limb = sum;
                    if !overflow {
                        return;
                    }
                }
                vec.push(1);
            }
        }
    }

    /// Subtracts 1 in-place, normalizing the result.
    ///
    /// # Panics
    /// Debug-mode only when called on zero.
    #[inline]
    pub fn decrement(&mut self) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                let old_len = usize::from(*len);
                // SAFETY: old_len <= INLINE_LIMBS, verified at construction
                let slice = unsafe { limbs.get_unchecked_mut(..old_len) };
                for (i, limb) in slice.iter_mut().enumerate() {
                    let (diff, underflow) = limb.overflowing_sub(1);
                    *limb = diff;
                    if !underflow {
                        // Borrow parou — trim só se o limb mais significativo ficou 0.
                        if diff == 0 && i.wrapping_add(1) == old_len {
                            *len = len.wrapping_sub(1);
                        }
                        return;
                    }
                }
                debug_assert!(false, "decrement of zero");
            }
            UintRepr::Heap(ref mut vec) => {
                let top = vec.len().wrapping_sub(1);
                for (i, limb) in vec.iter_mut().enumerate() {
                    let (diff, underflow) = limb.overflowing_sub(1);
                    *limb = diff;
                    if !underflow {
                        if diff == 0 && i == top {
                            vec.truncate(top);
                        }
                        return;
                    }
                }
                debug_assert!(false, "decrement of zero");
            }
        }
    }

    /// Returns a slice of the active limbs.
    #[inline]
    #[must_use]
    pub fn limbs(&self) -> &[Limb] {
        match self.repr {
            UintRepr::Inline { len, ref limbs } => {
                // SAFETY: len is guaranteed to be <= INLINE_LIMBS by construction.
                unsafe { limbs.get_unchecked(..usize::from(len)) }
            }
            UintRepr::Heap(ref vec) => vec.as_slice(),
        }
    }

    /// Returns a mutable slice of the active limbs.
    #[inline]
    #[must_use]
    pub fn limbs_mut(&mut self) -> &mut [Limb] {
        match self.repr {
            UintRepr::Inline { len, ref mut limbs } => {
                // SAFETY: len is guaranteed to be <= INLINE_LIMBS by construction.
                unsafe { limbs.get_unchecked_mut(..usize::from(len)) }
            }
            UintRepr::Heap(ref mut vec) => vec.as_mut_slice(),
        }
    }
}

#[cfg(test)]
#[path = "tests/storage.rs"]
mod tests;
