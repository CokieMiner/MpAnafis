//! Unsigned bitwise logic, rotations, bit reversal, and byte swapping.

#![expect(
    unsafe_code,
    reason = "Unchecked indices are bounded by min/max-derived loop limits or explicit index < slice.len() guards; raw destinations have capacity equal to their loop limit before set_len."
)]

use core::{
    cmp::{max, min},
    ptr::write_bytes,
};

use alloc::vec::Vec;

use super::{
    ArchKernels, BoundedPrecision, INLINE_LIMBS, InternalMpUint, LIMB_BITS, LIMB_BYTES, Limb,
};

impl InternalMpUint {
    /// Computes the bitwise AND of two unsigned integers.
    #[inline]
    #[must_use]
    pub fn bitand(&self, rhs: &Self) -> Self {
        // AND cannot retain limbs above the shorter operand, so one inline
        // input bounds the result to four limbs even when the other is large.
        if self.limbs().len() <= INLINE_LIMBS || rhs.limbs().len() <= INLINE_LIMBS {
            let [a0, a1, a2, a3] = self.extract_4();
            let [b0, b1, b2, b3] = rhs.extract_4();
            return Self::from_limbs_4(a0 & b0, a1 & b1, a2 & b2, a3 & b3);
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
        if self.limbs().len() <= INLINE_LIMBS && rhs.limbs().len() <= INLINE_LIMBS {
            let [a0, a1, a2, a3] = self.extract_4();
            let [b0, b1, b2, b3] = rhs.extract_4();
            return Self::from_limbs_4(a0 | b0, a1 | b1, a2 | b2, a3 | b3);
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
        // SAFETY: at least one normalized operand has a nonzero limb at
        // max_len-1. OR cannot clear it, so no normalization scan is needed.
        unsafe { Self::from_limbs_normalized(limbs) }
    }

    /// Computes the bitwise XOR of two unsigned integers.
    #[inline]
    #[must_use]
    pub fn bitxor(&self, rhs: &Self) -> Self {
        if self.limbs().len() <= INLINE_LIMBS && rhs.limbs().len() <= INLINE_LIMBS {
            let [a0, a1, a2, a3] = self.extract_4();
            let [b0, b1, b2, b3] = rhs.extract_4();
            return Self::from_limbs_4(a0 ^ b0, a1 ^ b1, a2 ^ b2, a3 ^ b3);
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
        if limb_count <= INLINE_LIMBS {
            let [a0, a1, a2, a3] = self.extract_4();
            return Self::from_limbs_4(!a0, !a1, !a2, !a3).apply_wrapping(width);
        }
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
        // SAFETY: shared <= limb_count; the vector reserves this complete
        // aligned span. All-ones bytes initialize valid limbs on every target,
        // and the suffix is disjoint from the initialized shared prefix.
        unsafe {
            write_bytes(dst.add(shared), 0xff, limb_count.unchecked_sub(shared));
        }
        // SAFETY: the complement loop and padding fill initialize every
        // result slot before the vector exposes its logical length.
        unsafe {
            limbs.set_len(limb_count);
        }

        if remaining_bits != 0 {
            // SAFETY: the nonzero remainder is below LIMB_BITS <= 64. The
            // difference lies in 1..LIMB_BITS on every supported pointer width.
            let mask = Limb::MAX >> unsafe { LIMB_BITS.unchecked_sub(remaining_bits) };
            // SAFETY: width > 0 proves limb_count > 0, so the initialized
            // result slice has a last limb under this exclusive borrow.
            *unsafe { limbs.last_mut().unwrap_unchecked() } &= mask;
        }

        if limb_count <= src_len {
            Self::from_limbs(limbs)
        } else {
            // SAFETY: the top limb is an all-ones extension; masking it to
            // a nonzero number of bits preserves a nonzero top limb.
            unsafe { Self::from_limbs_normalized(limbs) }
        }
    }

    /// Rotates the bits left by `n` positions within the given `width`.
    #[must_use]
    pub fn rotate_left(&self, n: u32, width: BoundedPrecision) -> Self {
        let rot = rotation_amount(n, width);
        let bits = width.get();
        let source = self.bit_range(0, bits);
        if rot == 0 {
            return source;
        }
        // SAFETY: rotation_amount reduces the full count modulo nonzero width,
        // so rot < width and the positive difference fits usize on every target.
        let shift_right = unsafe { bits.unchecked_sub(rot) };
        let left = source.shl(rot);
        let right = source.shr(shift_right);
        left.bitor(&right).apply_wrapping(bits)
    }

    /// Rotates the bits right by `n` positions within the given `width`.
    #[must_use]
    pub fn rotate_right(&self, n: u32, width: BoundedPrecision) -> Self {
        let rot = rotation_amount(n, width);
        let bits = width.get();
        let source = self.bit_range(0, bits);
        if rot == 0 {
            return source;
        }
        // SAFETY: rotation_amount reduces the full count modulo nonzero width,
        // so rot < width and the positive difference fits usize on every target.
        let shift_left = unsafe { bits.unchecked_sub(rot) };
        let right = source.shr(rot);
        let left = source.shl(shift_left);
        left.bitor(&right).apply_wrapping(bits)
    }

    /// Reverses the bit order within the given `width`.
    ///
    /// `width` must be non-zero.
    #[must_use]
    pub fn reverse_bits(&self, width: usize) -> Self {
        debug_assert!(width > 0, "bit reversal requires a non-zero width");
        if self.is_zero() {
            return Self::zero();
        }
        let src = self.limbs();
        let src_limb_count = src.len();
        let result_limb_count = width.div_ceil(LIMB_BITS);
        let rem = width.wrapping_rem(LIMB_BITS);
        // SAFETY: width > 0 proves result_limb_count > 0 on every pointer width.
        let top_index = unsafe { result_limb_count.unchecked_sub(1) };
        let total_shift = if rem == 0 {
            0
        } else {
            // SAFETY: 0 < rem < LIMB_BITS, so the padding lies in 1..LIMB_BITS.
            unsafe { LIMB_BITS.unchecked_sub(rem) }
        };
        // Reversal maps bits above width into the low total_shift padding of
        // the least significant output limb. The final right shift discards
        // that padding, so the original highest limb needs no separate mask.

        if result_limb_count <= INLINE_LIMBS {
            let mut inline = [0_usize; INLINE_LIMBS];
            for i in 0..result_limb_count {
                // SAFETY: i <= top_index by the loop range.
                let src_idx = unsafe { top_index.unchecked_sub(i) };
                let val = if src_idx < src_limb_count {
                    // SAFETY: src_idx < src_limb_count checked above
                    unsafe { *src.get_unchecked(src_idx) }
                } else {
                    0
                };
                // SAFETY: i < result_limb_count <= INLINE_LIMBS
                unsafe {
                    *inline.get_unchecked_mut(i) = val.reverse_bits();
                }
            }

            if total_shift > 0 {
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "total_shift < LIMB_BITS fits in u32"
                )]
                let shift_u32 = total_shift as u32;
                // SAFETY: inline has length INLINE_LIMBS >= result_limb_count, shift_u32 < LIMB_BITS
                unsafe {
                    let _ = ArchKernels::rshift_unchecked(
                        inline.as_mut_ptr(),
                        result_limb_count,
                        shift_u32,
                    );
                }
            }
            return Self::from_limbs_4(inline[0], inline[1], inline[2], inline[3]);
        }

        let mut result_limbs: Vec<Limb> = Vec::with_capacity(result_limb_count);
        let dst = result_limbs.as_mut_ptr();

        for i in 0..result_limb_count {
            // SAFETY: i <= top_index by the loop range.
            let src_idx = unsafe { top_index.unchecked_sub(i) };
            let val = if src_idx < src_limb_count {
                // SAFETY: src_idx < src_limb_count checked above
                unsafe { *src.get_unchecked(src_idx) }
            } else {
                0
            };
            // SAFETY: i < result_limb_count bounds the reserved aligned output;
            // every destination limb is initialized exactly once by this loop.
            unsafe {
                dst.add(i).write(val.reverse_bits());
            }
        }

        if total_shift > 0 {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "total_shift < LIMB_BITS fits in u32"
            )]
            let shift_u32 = total_shift as u32;
            // SAFETY: the preceding loop initialized all result_limb_count
            // reserved limbs, and 0 < shift_u32 < LIMB_BITS bounds the shift.
            unsafe {
                let _ = ArchKernels::rshift_unchecked(dst, result_limb_count, shift_u32);
            }
        }
        // SAFETY: the loop initialized the complete reserved prefix; the
        // optional in-place shift preserves initialization before length commit.
        unsafe {
            result_limbs.set_len(result_limb_count);
        }

        Self::from_limbs(result_limbs)
    }

    /// Reverses the low `ceil(width_bits / 8)` bytes with zero extension.
    ///
    /// An omitted width uses the significant bit count. A partial top byte
    /// participates in full; the operation does not mask that byte's high bits.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Byte offsets and their complementary shifts are below LIMB_BITS <= 64, fitting u32 on every supported target"
    )]
    pub fn swap_bytes(&self, width_bits: Option<usize>) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let sig = width_bits.unwrap_or_else(|| self.significant_bits());
        let byte_len = sig.div_ceil(8);
        if byte_len == 0 {
            return Self::zero();
        }
        let limbs = self.limbs();
        let result_limb_count = byte_len.div_ceil(LIMB_BYTES);
        // SAFETY: byte_len > 0 proves result_limb_count > 0.
        let top_index = unsafe { result_limb_count.unchecked_sub(1) };

        // If the original top limb was partial, the result is left-shifted by
        // (LIMB_BYTES - top_bytes) bytes. Compute the right-shift needed to
        // correct it (0 when all limbs are full).
        let top_bytes = byte_len.wrapping_rem(LIMB_BYTES);
        let shift_bits = if top_bytes != 0 {
            // SAFETY: 0 < top_bytes < LIMB_BYTES <= 8, so the difference
            // is in 1..=7 and its bit count is at most 56 on every target.
            (unsafe { LIMB_BYTES.unchecked_sub(top_bytes).unchecked_mul(8) }) as u32
        } else {
            0
        };
        // SAFETY: 0 <= shift_bits < LIMB_BITS <= 64. The complementary
        // shift is used only when shift_bits != 0, so it is then below LIMB_BITS.
        let carry_shift = unsafe { (LIMB_BITS as u32).unchecked_sub(shift_bits) };

        if result_limb_count <= INLINE_LIMBS {
            let mut inline = [0_usize; INLINE_LIMBS];
            let mut carry: Limb = 0;
            for i in 0..result_limb_count {
                let src_limb = limbs.get(i).copied().unwrap_or(0);
                // SAFETY: i <= top_index by the loop range.
                let dst = unsafe { top_index.unchecked_sub(i) };
                let swapped = src_limb.swap_bytes();
                // SAFETY: dst < result_limb_count <= INLINE_LIMBS
                unsafe {
                    *inline.get_unchecked_mut(dst) = swapped.wrapping_shr(shift_bits) | carry;
                }
                if shift_bits != 0 {
                    carry = swapped.wrapping_shl(carry_shift);
                }
            }
            return Self::from_limbs_4(inline[0], inline[1], inline[2], inline[3]);
        }

        let mut result_limbs: Vec<Limb> = Vec::with_capacity(result_limb_count);
        let output = result_limbs.as_mut_ptr();

        let mut carry: Limb = 0;
        for i in 0..result_limb_count {
            let src_limb = limbs.get(i).copied().unwrap_or(0);
            // SAFETY: i <= top_index by the loop range.
            let dst = unsafe { top_index.unchecked_sub(i) };
            let swapped = src_limb.swap_bytes();
            // SAFETY: dst = top_index - i < result_limb_count indexes the
            // reserved aligned output; every slot is initialized exactly once.
            unsafe {
                output
                    .add(dst)
                    .write(swapped.wrapping_shr(shift_bits) | carry);
            }
            if shift_bits != 0 {
                carry = swapped.wrapping_shl(carry_shift);
            }
        }
        // SAFETY: the descending output loop initialized the complete prefix.
        unsafe {
            result_limbs.set_len(result_limb_count);
        }

        Self::from_limbs(result_limbs)
    }
}

/// Reduces the full u32 count before converting the residue to a native index.
#[inline]
fn rotation_amount(n: u32, width: BoundedPrecision) -> usize {
    #[expect(
        clippy::as_conversions,
        reason = "usize is at most 64 bits on every supported target"
    )]
    let width_u64 = width.get() as u64;
    let remainder_u64 = u64::from(n).rem_euclid(width_u64);
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "the remainder is strictly smaller than the original usize width"
    )]
    let remainder = remainder_u64 as usize;
    remainder
}
