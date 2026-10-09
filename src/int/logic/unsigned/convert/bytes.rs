//! Byte-order conversions for the unsigned integer engine.

#![expect(
    unsafe_code,
    reason = "Exact byte and limb capacities bound raw writes; chunk iterators establish full-array lengths, and trimmed input proves normalization."
)]

use core::ptr::copy_nonoverlapping;

use alloc::vec::Vec;

use super::{InternalMpUint, LIMB_BYTES, Limb};

impl InternalMpUint {
    /// Returns the integer as a little-endian byte vector (least significant byte first).
    ///
    /// Leading zero bytes are not included. Returns an empty `Vec` for zero.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let limbs = self.limbs();
        let Some((&top, lower)) = limbs.split_last() else {
            return Vec::new();
        };
        debug_assert_ne!(top, 0, "a normalized magnitude has a nonzero highest limb");
        // SAFETY: top != 0 gives leading_zeros / 8 < LIMB_BYTES <= 8. The
        // conversion fits every usize and the difference lies in 1..=LIMB_BYTES.
        let top_len = unsafe {
            LIMB_BYTES.unchecked_sub(usize::try_from(top.leading_zeros() >> 3).unwrap_unchecked())
        };
        // SAFETY: lower.len() * LIMB_BYTES + top_len is at most the byte
        // size of the existing limb slice, bounded by isize::MAX. Both
        // intermediate results fit usize on 16-, 32-, and 64-bit targets.
        let lower_bytes = unsafe { lower.len().unchecked_mul(LIMB_BYTES) };
        // SAFETY: top_len <= LIMB_BYTES, so the total stays within that slice.
        let total_bytes = unsafe { lower_bytes.unchecked_add(top_len) };
        // SAFETY: the full byte width is the size of the existing limb slice,
        // so it is at most isize::MAX and fits usize on every pointer width.
        let capacity = unsafe { limbs.len().unchecked_mul(LIMB_BYTES) };
        let mut bytes: Vec<u8> = Vec::with_capacity(capacity);
        // A whole-limb copy avoids a runtime memcpy for the partial top limb.
        // At most LIMB_BYTES - 1 padding slots stay outside the exposed length.
        // SAFETY: the disjoint destination reserves every complete source limb.
        // All offsets are below capacity, and each exposed byte is initialized;
        // total_bytes commits exactly the normalized significant prefix.
        unsafe {
            let dst = bytes.as_mut_ptr();
            for (index, &limb) in limbs.iter().enumerate() {
                copy_nonoverlapping(
                    limb.to_le_bytes().as_ptr(),
                    dst.add(index.unchecked_mul(LIMB_BYTES)),
                    LIMB_BYTES,
                );
            }
            bytes.set_len(total_bytes);
        }
        bytes
    }

    /// Returns the integer as a big-endian byte vector (most significant byte first).
    ///
    /// Leading zero bytes are not included. Returns an empty `Vec` for zero.
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let Some((&top, lower)) = self.limbs().split_last() else {
            return Vec::new();
        };
        debug_assert_ne!(top, 0, "a normalized magnitude has a nonzero highest limb");
        // SAFETY: top != 0 gives leading_zeros / 8 < LIMB_BYTES <= 8, so the
        // padding count converts losslessly to usize on every pointer width.
        let skip = unsafe { usize::try_from(top.leading_zeros() >> 3).unwrap_unchecked() };
        // SAFETY: skip < LIMB_BYTES proves a positive top width in 1..=LIMB_BYTES.
        let top_len = unsafe { LIMB_BYTES.unchecked_sub(skip) };
        // SAFETY: this total is bounded by the existing limb slice's byte
        // size, which is at most isize::MAX on every supported pointer width.
        let total = unsafe { lower.len().unchecked_mul(LIMB_BYTES).unchecked_add(top_len) };
        let mut bytes: Vec<u8> = Vec::with_capacity(total);
        // SAFETY: skip < LIMB_BYTES, and top_len is the remaining initialized
        // top byte span. It and the reversed lower limbs fill exactly total
        // allocated slots. Local arrays and the destination cannot overlap.
        unsafe {
            let dst = bytes.as_mut_ptr();
            copy_nonoverlapping(top.to_be_bytes().as_ptr().add(skip), dst, top_len);
            for (index, &limb) in lower.iter().rev().enumerate() {
                copy_nonoverlapping(
                    limb.to_be_bytes().as_ptr(),
                    dst.add(top_len.unchecked_add(index.unchecked_mul(LIMB_BYTES))),
                    LIMB_BYTES,
                );
            }
            bytes.set_len(total);
        }
        bytes
    }

    /// Constructs an `InternalMpUint` from a little-endian byte slice.
    ///
    /// The bytes are interpreted as an unsigned integer in little-endian order
    /// (least significant byte first). An empty slice is treated as zero.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let significant_len = bytes
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |index| {
                // SAFETY: index < bytes.len(), so index + 1 <= bytes.len() fits usize.
                unsafe { index.unchecked_add(1) }
            });
        if significant_len == 0 {
            return Self::zero();
        }
        // SAFETY: significant_len is one past an index in bytes.
        let significant = unsafe { bytes.get_unchecked(..significant_len) };
        let num_limbs = significant_len.div_ceil(LIMB_BYTES);
        let mut result = Self::with_capacity(num_limbs);
        let mut write = result.prepare_limb_write(num_limbs);
        let dst = write.as_mut_ptr();
        let (chunks, tail) = significant.as_chunks::<LIMB_BYTES>();
        for (index, &array) in chunks.iter().enumerate() {
            // SAFETY: index < num_limbs; the guard owns aligned storage with
            // that capacity, and each iteration initializes a distinct limb.
            unsafe {
                dst.add(index).write(Limb::from_le_bytes(array));
            }
        }
        if !tail.is_empty() {
            let mut array = [0_u8; LIMB_BYTES];
            // SAFETY: the remainder is shorter than LIMB_BYTES; the final
            // index num_limbs - 1 is reserved and is not written by full chunks.
            unsafe {
                copy_nonoverlapping(tail.as_ptr(), array.as_mut_ptr(), tail.len());
                dst.add(num_limbs.unchecked_sub(1))
                    .write(Limb::from_le_bytes(array));
            }
        }
        // SAFETY: full chunks plus the optional partial chunk initialize
        // exactly num_limbs limbs. Trimming proves the top limb is nonzero.
        let _ = unsafe { write.commit() };
        result
    }

    /// Constructs an `InternalMpUint` from a big-endian byte slice.
    ///
    /// The bytes are interpreted as an unsigned integer in big-endian order
    /// (most significant byte first). An empty slice is treated as zero.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let first = bytes
            .iter()
            .position(|&byte| byte != 0)
            .unwrap_or(bytes.len());
        // SAFETY: first is an index in bytes or its length.
        let significant = unsafe { bytes.get_unchecked(first..) };
        if significant.is_empty() {
            return Self::zero();
        }
        let num_limbs = significant.len().div_ceil(LIMB_BYTES);
        let mut result = Self::with_capacity(num_limbs);
        let mut write = result.prepare_limb_write(num_limbs);
        let dst = write.as_mut_ptr();
        let (head, chunks) = significant.as_rchunks::<LIMB_BYTES>();
        for (index, &array) in chunks.iter().rev().enumerate() {
            // SAFETY: index < num_limbs; the guard owns aligned storage with
            // that capacity, and each iteration initializes a distinct limb.
            unsafe {
                dst.add(index).write(Limb::from_be_bytes(array));
            }
        }
        if !head.is_empty() {
            let mut array = [0_u8; LIMB_BYTES];
            // SAFETY: head.len() < LIMB_BYTES, so the right-aligned copy fits
            // the local array. The final reserved limb follows all full chunks.
            unsafe {
                copy_nonoverlapping(
                    head.as_ptr(),
                    array.as_mut_ptr().add(LIMB_BYTES.unchecked_sub(head.len())),
                    head.len(),
                );
                dst.add(num_limbs.unchecked_sub(1))
                    .write(Limb::from_be_bytes(array));
            }
        }
        // SAFETY: full chunks plus the optional partial chunk initialize
        // exactly num_limbs limbs. Trimming proves the top limb is nonzero.
        let _ = unsafe { write.commit() };
        result
    }
}
