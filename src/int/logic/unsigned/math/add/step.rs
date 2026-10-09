//! Unit increments and decrements with native limb carry and borrow propagation.

#![expect(
    unsafe_code,
    reason = "Canonical storage bounds inline access and guarantees termination of nonzero decrement borrows"
)]

use core::ptr::copy_nonoverlapping;

use alloc::vec::Vec;

use super::{INLINE_LIMBS, InternalMpUint, UintRepr};

impl InternalMpUint {
    /// Adds one in place, extending the magnitude when the carry reaches its end.
    #[inline]
    pub fn increment(&mut self) {
        match self.repr {
            UintRepr::Inline {
                ref mut len,
                ref mut limbs,
            } => {
                let old_len = usize::from(*len);
                // SAFETY: the representation bounds old_len by INLINE_LIMBS,
                // and the entire active prefix is initialized and exclusively borrowed.
                let slice = unsafe { limbs.get_unchecked_mut(..old_len) };
                for limb in slice {
                    let (sum, overflow) = limb.overflowing_add(1);
                    *limb = sum;
                    if !overflow {
                        return;
                    }
                }
                if old_len < INLINE_LIMBS {
                    // SAFETY: old_len < INLINE_LIMBS bounds the initialized
                    // destination slot; len + 1 <= INLINE_LIMBS = 4 fits u8.
                    unsafe {
                        *limbs.get_unchecked_mut(old_len) = 1;
                        *len = len.unchecked_add(1);
                    }
                } else {
                    let mut vec = Vec::with_capacity(INLINE_LIMBS + 1);
                    // SAFETY: old_len = INLINE_LIMBS. The fresh, aligned allocation
                    // is disjoint and reserves five limbs; the copy and final write
                    // initialize every slot before its length is committed.
                    unsafe {
                        copy_nonoverlapping(limbs.as_ptr(), vec.as_mut_ptr(), INLINE_LIMBS);
                        vec.as_mut_ptr().add(INLINE_LIMBS).write(1);
                        vec.set_len(INLINE_LIMBS + 1);
                    }
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

    /// Subtracts one from a caller-proved nonzero normalized magnitude.
    ///
    /// Borrow propagates through the lower prefix. The nonzero highest limb
    /// absorbs any remaining borrow, and only that column can shorten the value.
    ///
    /// # Panics
    /// Debug-mode only when called on zero or an unnormalized magnitude.
    #[inline]
    pub fn decrement(&mut self) {
        let span = self.limbs_mut();
        debug_assert!(!span.is_empty(), "decrement requires a non-zero value");
        // SAFETY: callers establish a nonzero canonical magnitude before
        // decrement. Its initialized active slice is nonempty, so splitting
        // off the highest limb cannot fail. Unchecked extraction keeps that
        // proved precondition from becoming an extra release-mode branch.
        let (top, lower) = unsafe { span.split_last_mut().unwrap_unchecked() };
        debug_assert_ne!(*top, 0, "a normalized magnitude ends in a nonzero limb");

        // Borrow can terminate in the lower prefix without changing the length.
        // Zero lower limbs produce MAX and pass the borrow to the next column.
        for limb in lower {
            let (difference, underflow) = limb.overflowing_sub(1);
            *limb = difference;
            if !underflow {
                return;
            }
        }
        // SAFETY: normalization proves top > 0. Lower-column writes are
        // disjoint from top, so top - 1 is representable on every limb width.
        *top = unsafe { top.unchecked_sub(1) };
        if *top == 0 {
            match self.repr {
                UintRepr::Inline { ref mut len, .. } => {
                    // SAFETY: the nonempty active slice proves 1 <= len <= 4;
                    // no lower-column write changes the encoded length.
                    *len = unsafe { len.unchecked_sub(1) };
                }
                UintRepr::Heap(ref mut limbs) => {
                    // SAFETY: the initial active slice was nonempty. Limb
                    // writes leave its Vec length unchanged, so len - 1 fits
                    // and retains an initialized prefix within capacity.
                    unsafe {
                        limbs.set_len(limbs.len().unchecked_sub(1));
                    }
                }
            }
        }
    }
}
