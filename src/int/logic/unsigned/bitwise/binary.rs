//! Unsigned bitwise logic, rotations, bit reversal, and byte swapping.

#![allow(
    unsafe_code,
    reason = "Unchecked indices are bounded by min/max-derived loop limits or explicit index < slice.len() guards; raw destinations have capacity equal to their loop limit before set_len."
)]

use core::cmp::{max, min};

use alloc::vec::Vec;

use super::{ArchKernels, INLINE_LIMBS, InternalArbiUint, LIMB_BITS, LIMB_BYTES, Limb, UintRepr};

impl InternalArbiUint {
    /// Computes the bitwise AND of two unsigned integers.
    #[inline]
    #[must_use]
    pub fn bitand(&self, rhs: &Self) -> Self {
        if let (UintRepr::Inline { len: l1, limbs: a }, UintRepr::Inline { len: l2, limbs: b }) =
            (&self.repr, &rhs.repr)
        {
            let min_len = min(*l1, *l2);
            let mut arr = [0; INLINE_LIMBS];
            for (i, v) in arr.iter_mut().enumerate().take(usize::from(min_len)) {
                // SAFETY: i < min_len <= INLINE_LIMBS, and both inline arrays
                // have exactly INLINE_LIMBS elements.
                *v = unsafe { a.get_unchecked(i) & b.get_unchecked(i) };
            }
            let mut res = Self {
                repr: UintRepr::Inline {
                    len: min_len,
                    limbs: arr,
                },
            };
            res.normalize();
            return res;
        }

        let a = self.limbs();
        let b = rhs.limbs();
        let min_len = min(a.len(), b.len());
        let mut limbs: Vec<Limb> = Vec::with_capacity(min_len);
        let dst = limbs.as_mut_ptr();

        for i in 0..min_len {
            // SAFETY: `dst` points to an allocation with capacity `min_len`.
            // For every i in 0..min_len, dst.add(i) is within that allocation.
            // The source indices are valid because min_len <= a.len(), b.len().
            // Each destination slot is written exactly once before set_len.
            unsafe {
                dst.add(i).write(*a.get_unchecked(i) & *b.get_unchecked(i));
            }
        }
        // SAFETY: The loop above initialized exactly the first `min_len` slots.
        unsafe {
            limbs.set_len(min_len);
        }
        Self::from_limbs(limbs)
    }

    /// Computes the bitwise OR of two unsigned integers.
    #[inline]
    #[must_use]
    pub fn bitor(&self, rhs: &Self) -> Self {
        if let (UintRepr::Inline { len: l1, limbs: a }, UintRepr::Inline { len: l2, limbs: b }) =
            (&self.repr, &rhs.repr)
        {
            let max_len = max(*l1, *l2);
            let mut arr = [0; INLINE_LIMBS];
            for (i, v) in arr.iter_mut().enumerate().take(usize::from(max_len)) {
                let a_val = if i < usize::from(*l1) {
                    // SAFETY: i < l1 <= INLINE_LIMBS, so the index is within the array bounds
                    unsafe { *a.get_unchecked(i) }
                } else {
                    0
                };
                let b_val = if i < usize::from(*l2) {
                    // SAFETY: i < l2 <= INLINE_LIMBS, so the index is within the array bounds
                    unsafe { *b.get_unchecked(i) }
                } else {
                    0
                };
                *v = a_val | b_val;
            }
            let mut res = Self {
                repr: UintRepr::Inline {
                    len: max_len,
                    limbs: arr,
                },
            };
            res.normalize();
            return res;
        }

        let a = self.limbs();
        let b = rhs.limbs();
        let a_len = a.len();
        let b_len = b.len();
        let max_len = max(a_len, b_len);
        let min_len = min(a_len, b_len);
        let mut limbs: Vec<Limb> = Vec::with_capacity(max_len);
        let dst = limbs.as_mut_ptr();

        for i in 0..min_len {
            // SAFETY: `dst` points to an allocation with capacity `max_len`.
            // For every i in 0..min_len, dst.add(i) is within that allocation.
            // The source indices are valid because min_len <= a_len, b_len.
            // Each destination slot is written exactly once before set_len.
            unsafe {
                dst.add(i).write(*a.get_unchecked(i) | *b.get_unchecked(i));
            }
        }
        let (long, long_len) = if a_len >= b_len {
            (a, a_len)
        } else {
            (b, b_len)
        };
        for i in min_len..long_len {
            // SAFETY: i < long_len <= max_len (capacity), and i < long.len().
            unsafe {
                dst.add(i).write(*long.get_unchecked(i));
            }
        }
        // SAFETY: All max_len slots have been initialized.
        unsafe {
            limbs.set_len(max_len);
        }
        Self::from_limbs(limbs)
    }

    /// Computes the bitwise XOR of two unsigned integers.
    #[inline]
    #[must_use]
    pub fn bitxor(&self, rhs: &Self) -> Self {
        if let (UintRepr::Inline { len: l1, limbs: a }, UintRepr::Inline { len: l2, limbs: b }) =
            (&self.repr, &rhs.repr)
        {
            let max_len = max(*l1, *l2);
            let mut arr = [0; INLINE_LIMBS];
            for (i, v) in arr.iter_mut().enumerate().take(usize::from(max_len)) {
                let a_val = if i < usize::from(*l1) {
                    // SAFETY: i < l1 <= INLINE_LIMBS, so the index is within the array bounds
                    unsafe { *a.get_unchecked(i) }
                } else {
                    0
                };
                let b_val = if i < usize::from(*l2) {
                    // SAFETY: i < l2 <= INLINE_LIMBS, so the index is within the array bounds
                    unsafe { *b.get_unchecked(i) }
                } else {
                    0
                };
                *v = a_val ^ b_val;
            }
            let mut res = Self {
                repr: UintRepr::Inline {
                    len: max_len,
                    limbs: arr,
                },
            };
            res.normalize();
            return res;
        }

        let a = self.limbs();
        let b = rhs.limbs();
        let a_len = a.len();
        let b_len = b.len();
        let max_len = max(a_len, b_len);
        let min_len = min(a_len, b_len);
        let mut limbs: Vec<Limb> = Vec::with_capacity(max_len);
        let dst = limbs.as_mut_ptr();

        for i in 0..min_len {
            // SAFETY: `dst` points to an allocation with capacity `max_len`.
            // For every i in 0..min_len, dst.add(i) is within that allocation.
            // The source indices are valid because min_len <= a_len, b_len.
            // Each destination slot is written exactly once before set_len.
            unsafe {
                dst.add(i).write(*a.get_unchecked(i) ^ *b.get_unchecked(i));
            }
        }
        let (long, long_len) = if a_len >= b_len {
            (a, a_len)
        } else {
            (b, b_len)
        };
        for i in min_len..long_len {
            // SAFETY: i < long_len <= max_len (capacity), and i < long.len().
            unsafe {
                dst.add(i).write(*long.get_unchecked(i));
            }
        }
        // SAFETY: All max_len slots have been initialized.
        unsafe {
            limbs.set_len(max_len);
        }
        Self::from_limbs(limbs)
    }

    /// Computes the bitwise NOT within an explicit bit width.
    ///
    /// `width` must be non-zero.
    #[must_use]
    pub fn not(&self, width: usize) -> Self {
        debug_assert!(width > 0, "bitwise NOT requires a non-zero width");

        let limb_count = width.div_ceil(LIMB_BITS);
        let remaining_bits = width.wrapping_rem(LIMB_BITS);

        let src = self.limbs();
        let src_len = src.len();
        let mut limbs: Vec<Limb> = Vec::with_capacity(limb_count);
        let dst = limbs.as_mut_ptr();

        // Split loop: shared limbs (direct NOT) — no per-iteration branch.
        let shared = min(src_len, limb_count);
        for i in 0..shared {
            // SAFETY: `dst` points to an allocation with capacity `limb_count`.
            // For every i in 0..shared, dst.add(i) is within that allocation.
            // The source index is valid because shared <= src_len.
            // Each destination slot is written exactly once before set_len.
            unsafe {
                dst.add(i).write(!*src.get_unchecked(i));
            }
        }
        // Padding limbs: NOT of zero = all-ones.
        for i in shared..limb_count {
            // SAFETY: `dst` points to an allocation with capacity `limb_count`.
            // For every i in shared..limb_count, dst.add(i) is within that allocation.
            // These slots are disjoint from the shared loop and are written once.
            unsafe {
                dst.add(i).write(!0);
            }
        }
        // SAFETY: The two loops above initialized exactly the first `limb_count` slots.
        unsafe {
            limbs.set_len(limb_count);
        }

        if remaining_bits != 0 {
            let mask = low_bits_mask(remaining_bits);
            if let Some(last) = limbs.last_mut() {
                *last &= mask;
            }
        }

        Self::from_limbs(limbs)
    }

    /// Masks the integer to the lower `width` bits.
    ///
    /// `width` must be non-zero.
    ///
    /// This replaces the double-`not(width)` pattern (which allocates twice)
    /// with a single direct masking pass.
    #[must_use]
    pub fn mask_to_width(&self, width: usize) -> Self {
        debug_assert!(width > 0, "bit masking requires a non-zero width");

        let limb_count = width.div_ceil(LIMB_BITS);
        let remaining_bits = width.wrapping_rem(LIMB_BITS);

        let src = self.limbs();
        let src_len = src.len();
        let copy_len = min(src_len, limb_count);
        let mut limbs = Vec::with_capacity(limb_count);

        // Copy (already masked by length) limbs directly using memcpy.
        // SAFETY: `copy_len = min(src.len(), limb_count)`, so the range
        // `..copy_len` is within the initialized shared source slice.
        limbs.extend_from_slice(unsafe { src.get_unchecked(..copy_len) });

        // Pad with zeros if the source is shorter using memset.
        limbs.resize(limb_count, 0);

        // Mask the top limb if the width is not a multiple of LIMB_BITS.
        if remaining_bits != 0 {
            let mask = low_bits_mask(remaining_bits);
            if let Some(last) = limbs.last_mut() {
                *last &= mask;
            }
        }

        Self::from_limbs(limbs)
    }

    /// Rotates the bits left by `n` positions within the given `width`.
    ///
    /// `width` must be non-zero.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        reason = "u32 rotate count: truncated cast is safe — result only used modulo width. Bypasses checked conversions to keep code branchless."
    )]
    pub fn rotate_left(&self, n: u32, width: usize) -> Self {
        debug_assert!(width > 0, "bit rotation requires a non-zero width");

        let n_usize = n as usize;
        // SAFETY: the public boundary rejects zero width.
        let rot = unsafe { n_usize.checked_rem(width).unwrap_unchecked() };
        if rot == 0 {
            return self.mask_to_width(width);
        }
        let shift_right = width.wrapping_sub(rot);
        let mut res = self.shl(rot);
        let right = self.shr(shift_right);
        let dst_len = res.limbs().len();
        for (i, &val) in right.limbs().iter().enumerate() {
            if i < dst_len {
                // SAFETY: i < dst_len == res.limbs().len()
                unsafe {
                    *res.limbs_mut().get_unchecked_mut(i) |= val;
                }
            } else if let UintRepr::Heap(ref mut vec) = res.repr {
                vec.push(val);
            }
        }
        res.normalize();
        res.mask_to_width(width)
    }

    /// Rotates the bits right by `n` positions within the given `width`.
    ///
    /// `width` must be non-zero.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        reason = "u32 rotate count: truncated cast is safe — result only used modulo width. Bypasses checked conversions to keep code branchless."
    )]
    pub fn rotate_right(&self, n: u32, width: usize) -> Self {
        debug_assert!(width > 0, "bit rotation requires a non-zero width");

        let n_usize = n as usize;
        // SAFETY: the public boundary rejects zero width.
        let rot = unsafe { n_usize.checked_rem(width).unwrap_unchecked() };
        if rot == 0 {
            return self.mask_to_width(width);
        }
        let shift_left = width.wrapping_sub(rot);
        let right = self.shr(rot);
        let mut res = self.shl(shift_left);
        let dst_len = res.limbs().len();
        for (i, &val) in right.limbs().iter().enumerate() {
            if i < dst_len {
                // SAFETY: i < dst_len == res.limbs().len()
                unsafe {
                    *res.limbs_mut().get_unchecked_mut(i) |= val;
                }
            } else if let UintRepr::Heap(ref mut vec) = res.repr {
                vec.push(val);
            }
        }
        res.normalize();
        res.mask_to_width(width)
    }

    /// Reverses the bit order within the given `width`.
    ///
    /// `width` must be non-zero.
    #[must_use]
    pub fn reverse_bits(&self, width: usize) -> Self {
        debug_assert!(width > 0, "bit reversal requires a non-zero width");
        let src = self.limbs();
        let src_limb_count = src.len();
        let result_limb_count = width.div_ceil(LIMB_BITS);

        let mut result_limbs = alloc::vec![0; result_limb_count];
        let rem = width.wrapping_rem(LIMB_BITS);

        for i in 0..result_limb_count {
            let src_idx = result_limb_count.wrapping_sub(1).wrapping_sub(i);
            let mut val = if src_idx < src_limb_count {
                // SAFETY: src_idx < src_limb_count checked above
                unsafe { *src.get_unchecked(src_idx) }
            } else {
                0
            };
            if src_idx == result_limb_count.wrapping_sub(1) && rem != 0 {
                val &= low_bits_mask(rem);
            }
            // SAFETY: i < result_limb_count by loop bounds
            unsafe {
                *result_limbs.get_unchecked_mut(i) = val.reverse_bits();
            }
        }

        let total_shift = result_limb_count
            .wrapping_mul(LIMB_BITS)
            .wrapping_sub(width);
        if total_shift > 0 {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "total_shift < LIMB_BITS fits in u32"
            )]
            let shift_u32 = total_shift as u32;
            // SAFETY: result_limbs has length result_limb_count, shift_u32 < LIMB_BITS
            unsafe {
                let _ = ArchKernels::rshift_unchecked(
                    result_limbs.as_mut_ptr(),
                    result_limb_count,
                    shift_u32,
                );
            }
        }

        Self::from_limbs(result_limbs)
    }

    /// Swaps the byte order of the integer value.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "byte idx * 8 fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
    )]
    pub fn swap_bytes(&self, width_bits: Option<usize>) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let sig = width_bits.unwrap_or_else(|| self.significant_bits());
        let byte_len = sig.wrapping_add(7).wrapping_div(8);
        if byte_len == 0 {
            return Self::zero();
        }
        let limbs = self.limbs();
        let result_limb_count = byte_len
            .wrapping_add(LIMB_BYTES.wrapping_sub(1))
            .wrapping_div(LIMB_BYTES);

        let mut result_limbs = alloc::vec![0; result_limb_count];

        // If the original top limb was partial, the result is left-shifted by
        // (LIMB_BYTES - top_bytes) bytes. Compute the right-shift needed to
        // correct it (0 when all limbs are full).
        let top_bytes = byte_len.wrapping_rem(LIMB_BYTES);
        let shift_bits = if top_bytes != 0 {
            (LIMB_BYTES.wrapping_sub(top_bytes).wrapping_mul(8)) as u32
        } else {
            0
        };

        let mut carry: Limb = 0;
        for i in 0..result_limb_count {
            let src_limb = limbs.get(i).copied().unwrap_or(0);
            let dst = result_limb_count.wrapping_sub(1).wrapping_sub(i);
            let swapped = src_limb.swap_bytes();
            // SAFETY: dst < result_limb_count by loop construction (i < result_limb_count).
            unsafe {
                *result_limbs.get_unchecked_mut(dst) = swapped.wrapping_shr(shift_bits) | carry;
            }
            if shift_bits != 0 {
                carry = swapped.wrapping_shl((LIMB_BITS as u32).wrapping_sub(shift_bits));
            }
        }

        Self::from_limbs(result_limbs)
    }
}

#[inline]
fn low_bits_mask(bits: usize) -> Limb {
    debug_assert!(bits <= LIMB_BITS, "bits must be <= LIMB_BITS");
    if bits == 0 {
        0
    } else if bits == LIMB_BITS {
        Limb::MAX
    } else {
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64, difference fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
        )]
        Limb::MAX.wrapping_shr(LIMB_BITS.wrapping_sub(bits) as u32)
    }
}

#[cfg(test)]
#[path = "tests/binary.rs"]
mod tests;
