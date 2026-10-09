//! Unsigned integer storage representation and canonical limb management.

#![expect(
    unsafe_code,
    reason = "Representation bounds and initialized storage prove raw copies, slice construction, and length updates"
)]

use core::ptr::copy_nonoverlapping;

use alloc::vec::Vec;

use super::{INLINE_LIMBS, Limb};

/// The internal representation of the magnitude.
///
/// Inline arrays are fully initialized, with `len <= INLINE_LIMBS = 4`.
/// Inactive slots are unspecified. The active prefix is normalized except
/// during explicit resizing or raw arithmetic, whose caller must normalize
/// before magnitude queries. Heap values may retain capacity at any length.
#[derive(Debug, PartialEq, Eq, Hash)]
#[doc(hidden)]
#[non_exhaustive]
pub enum UintRepr {
    /// Four initialized limbs with an independently tracked active prefix.
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
                dest_vec.clone_from(src_vec); // Reuses pre-allocated heap buffer capacity.
            }
            (Self::Heap(dest_vec), Self::Inline { len, limbs }) => {
                dest_vec.clear();
                // SAFETY: usize::from(*len) <= INLINE_LIMBS, so ..usize::from(*len) is within the limbs array bounds.
                let slice = unsafe { limbs.get_unchecked(..usize::from(*len)) };
                dest_vec.extend_from_slice(slice);
            }
            (dest, src) => {
                *dest = src.clone();
            }
        }
    }
}

/// The core unsigned arbitrary precision integer engine.
#[derive(Debug)]
pub struct InternalMpUint {
    pub repr: UintRepr,
}

impl InternalMpUint {
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

    /// Creates a new unsigned integer from up to 2 limbs.
    #[inline]
    #[must_use]
    pub const fn from_limbs_2(lo: Limb, hi: Limb) -> Self {
        if hi == 0 {
            Self::from_limb(lo)
        } else {
            Self {
                repr: UintRepr::Inline {
                    len: 2,
                    limbs: [lo, hi, 0, 0],
                },
            }
        }
    }

    /// Creates a new unsigned integer from up to 4 limbs.
    #[inline]
    #[must_use]
    pub const fn from_limbs_4(l0: Limb, l1: Limb, l2: Limb, l3: Limb) -> Self {
        if l3 != 0 {
            Self {
                repr: UintRepr::Inline {
                    len: 4,
                    limbs: [l0, l1, l2, l3],
                },
            }
        } else if l2 != 0 {
            Self {
                repr: UintRepr::Inline {
                    len: 3,
                    limbs: [l0, l1, l2, 0],
                },
            }
        } else {
            Self::from_limbs_2(l0, l1)
        }
    }

    /// Extracts the lowest 4 limbs into a fixed array `[Limb; 4]`.
    /// Zero-pads if the integer has fewer than 4 limbs.
    #[expect(
        clippy::inline_always,
        reason = "Inlining exposes the fixed four-limb extraction to scalar arithmetic callers and permits their unused output lanes to be removed"
    )]
    #[inline(always)]
    #[must_use]
    pub fn extract_4(&self) -> [Limb; 4] {
        match self.repr {
            UintRepr::Inline { len, limbs } => match len {
                0 => [0, 0, 0, 0],
                1 => [limbs[0], 0, 0, 0],
                2 => [limbs[0], limbs[1], 0, 0],
                3 => [limbs[0], limbs[1], limbs[2], 0],
                _ => limbs,
            },
            UintRepr::Heap(ref v) => {
                let mut res = [0; 4];
                let len = v.len().min(4);
                if len > 0 {
                    // SAFETY: len <= v.len() and len <= res.len(); both pointers
                    // are aligned. The initialized source and fresh destination
                    // cannot overlap. Empty heap values need no copy.
                    unsafe {
                        copy_nonoverlapping(v.as_ptr(), res.as_mut_ptr(), len);
                    }
                }
                res
            }
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
    #[expect(
        clippy::inline_always,
        reason = "Inlining exposes the four-limb storage boundary to callers that prove their normalized output width"
    )]
    #[inline(always)]
    #[must_use]
    pub unsafe fn from_limbs_normalized(limbs: Vec<Limb>) -> Self {
        debug_assert!(
            limbs.last().is_none_or(|&x| x != 0),
            "normalized inputs must not end with zero"
        );
        let len = limbs.len();
        if len <= INLINE_LIMBS {
            let mut arr = [0; INLINE_LIMBS];
            // SAFETY: len <= INLINE_LIMBS bounds the initialized source and
            // fresh disjoint array. Copying fills every active destination slot.
            unsafe {
                copy_nonoverlapping(limbs.as_ptr(), arr.as_mut_ptr(), len);
            }
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "inline limb count is at most INLINE_LIMBS — always fits in u8"
            )]
            let len_u8 = len as u8;
            Self {
                repr: UintRepr::Inline {
                    len: len_u8,
                    limbs: arr,
                },
            }
        } else {
            Self {
                repr: UintRepr::Heap(limbs),
            }
        }
    }

    /// Creates an integer from an arbitrary limb slice, trimming high zero limbs.
    #[expect(
        clippy::inline_always,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Inlining exposes fixed source widths to storage selection; the inline branch bounds its length by four, fitting u8 on every target"
    )]
    #[inline(always)]
    #[must_use]
    pub fn from_limbs_slice(limbs: &[Limb]) -> Self {
        let len = normalized_len(limbs);
        // SAFETY: normalized_len returns a prefix length at most limbs.len().
        let trimmed = unsafe { limbs.get_unchecked(..len) };
        if len <= INLINE_LIMBS {
            let mut arr = [0; INLINE_LIMBS];
            // SAFETY: this branch proves `len <= INLINE_LIMBS == arr.len()`.
            unsafe { arr.get_unchecked_mut(..len) }.copy_from_slice(trimmed);
            Self {
                repr: UintRepr::Inline {
                    len: len as u8,
                    limbs: arr,
                },
            }
        } else {
            Self {
                repr: UintRepr::Heap(trimmed.to_vec()),
            }
        }
    }

    /// Creates an integer from an owned limb vector.
    ///
    /// Trims any trailing zero limbs so that the internal representation is normalized.
    #[expect(
        clippy::inline_always,
        reason = "from_limbs is a core constructor called on every operation result"
    )]
    #[inline(always)]
    #[must_use]
    pub fn from_limbs(mut limbs: Vec<Limb>) -> Self {
        let len = normalized_len(&limbs);
        // SAFETY: len <= the initialized vector length; discarded Limb values
        // have no destructor. The retained prefix ends in a nonzero limb or is empty.
        unsafe {
            limbs.set_len(len);
        }
        // SAFETY: normalized_len removed every high zero limb above.
        unsafe { Self::from_limbs_normalized(limbs) }
    }

    /// Creates an integer from a single limb.
    #[inline]
    #[must_use]
    pub const fn from_limb(limb: Limb) -> Self {
        if limb == 0 {
            Self::zero()
        } else {
            Self {
                repr: UintRepr::Inline {
                    len: 1,
                    limbs: [limb, 0, 0, 0],
                },
            }
        }
    }
    /// Replaces this value with a single limb.
    #[inline]
    pub fn set_limb(&mut self, limb: Limb) {
        if limb == 0 {
            self.clear();
            return;
        }
        match self.repr {
            UintRepr::Heap(ref mut vec) => {
                vec.clear();
                vec.push(limb);
            }
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                *len = 1;
                limbs[0] = limb;
            }
        }
    }

    /// Replaces this value with the normalized limbs from `slice`.
    pub fn clone_from_slice(&mut self, slice: &[Limb]) {
        let len = normalized_len(slice);
        if len == 0 {
            self.clear();
            return;
        }
        // SAFETY: normalized_len returns a prefix length at most slice.len().
        let trimmed_slice = unsafe { slice.get_unchecked(..len) };
        match self.repr {
            UintRepr::Heap(ref mut vec) => {
                vec.clear();
                vec.extend_from_slice(trimmed_slice);
            }
            UintRepr::Inline {
                len: ref mut inline_len,
                limbs: ref mut inline_limbs,
            } => {
                if len <= INLINE_LIMBS {
                    // SAFETY: len <= INLINE_LIMBS and len <= slice.len(); the
                    // mutable destination cannot alias the borrowed source.
                    // Inactive inline slots lie outside the logical value.
                    unsafe {
                        copy_nonoverlapping(slice.as_ptr(), inline_limbs.as_mut_ptr(), len);
                    }
                    #[expect(
                        clippy::as_conversions,
                        clippy::cast_possible_truncation,
                        reason = "inline limb count is at most INLINE_LIMBS — always fits in u8"
                    )]
                    let len_u8 = len as u8;
                    *inline_len = len_u8;
                } else {
                    self.repr = UintRepr::Heap(trimmed_slice.to_vec());
                }
            }
        }
    }

    /// Sets the length of the internal representation without initializing the memory.
    /// Normalize before magnitude queries if the new highest limb is zero.
    ///
    /// # Safety
    /// `new_len` must not exceed capacity, and every element in `0..new_len`
    /// must already be initialized before this call.
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

    /// Removes high zero limbs while retaining the storage allocation.
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "Inlining exposes caller-known inline lengths and top-limb values to the normalization scan"
    )]
    pub fn normalize(&mut self) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                // SAFETY: the representation bounds len by INLINE_LIMBS, and
                // every inline slot is initialized, including arithmetic zeroes.
                let active = unsafe { limbs.get_unchecked(..usize::from(*len)) };
                let new_len = normalized_len(active);
                // SAFETY: new_len <= the previous inline length <= 4 fits u8.
                *len = unsafe { u8::try_from(new_len).unwrap_unchecked() };
            }
            UintRepr::Heap(ref mut vec) => {
                let new_len = normalized_len(vec);
                // SAFETY: new_len <= the initialized old length <= capacity,
                // and discarded Limb values have no destructor.
                unsafe {
                    vec.set_len(new_len);
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

/// Returns a prefix length at most `limbs.len()`, ending at the highest nonzero limb.
#[inline]
fn normalized_len(limbs: &[Limb]) -> usize {
    let mut len = limbs.len();
    while len > 0 {
        // SAFETY: len starts at limbs.len() and only decreases; len > 0
        // proves subtraction cannot underflow on any pointer width.
        let last = unsafe { len.unchecked_sub(1) };
        // SAFETY: last < len <= limbs.len(), so this initialized limb is in bounds.
        if unsafe { *limbs.get_unchecked(last) } != 0 {
            break;
        }
        len = last;
    }
    len
}
