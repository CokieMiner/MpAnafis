//! Byte-order conversions for the unsigned integer engine.

#![allow(
    unsafe_code,
    reason = "Byte writes cover exactly limbs.len() * LIMB_BYTES allocated slots; unchecked ranges are bounded by position results, chunks of at most LIMB_BYTES, or start <= pos <= bytes.len()."
)]
use core::ptr::copy_nonoverlapping;

use alloc::vec::Vec;

use super::{InternalArbiUint, LIMB_BYTES, Limb};
impl InternalArbiUint {
    /// Returns the integer as a little-endian byte vector (least significant byte first).
    ///
    /// Leading zero bytes are not included. Returns an empty `Vec` for zero.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let limbs = self.limbs();
        let total_bytes = limbs.len().wrapping_mul(LIMB_BYTES);
        let mut bytes: Vec<u8> = Vec::with_capacity(total_bytes);

        // SAFETY: We pre-allocate `total_bytes` capacity above, then copy each limb's
        // little-endian bytes into the buffer. The write pointer advances by LIMB_BYTES
        // per limb, staying within the allocated capacity.
        unsafe {
            let dst = bytes.as_mut_ptr();
            for (i, &limb) in limbs.iter().enumerate() {
                let limb_bytes: [u8; LIMB_BYTES] = limb.to_le_bytes();
                copy_nonoverlapping(
                    limb_bytes.as_ptr(),
                    dst.add(i.wrapping_mul(LIMB_BYTES)),
                    LIMB_BYTES,
                );
            }
            bytes.set_len(total_bytes);
        }

        // Strip trailing zero bytes (canonical representation).
        if let Some(pos) = bytes.iter().rposition(|&b| b != 0) {
            bytes.truncate(pos.wrapping_add(1));
        } else {
            bytes.clear();
        }

        bytes
    }

    /// Returns the integer as a big-endian byte vector (most significant byte first).
    ///
    /// Leading zero bytes are not included. Returns an empty `Vec` for zero.
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let limbs = self.limbs();
        let Some((&top_limb, lower_limbs)) = limbs.split_last() else {
            return Vec::new();
        };
        debug_assert!(top_limb != 0, "InternalArbiUint limbs must be normalized");
        let top_be: [u8; LIMB_BYTES] = top_limb.to_be_bytes();
        let first_non_zero = top_be.iter().position(|&b| b != 0).unwrap_or(0);
        // SAFETY: `position` returns an index below `top_be.len()`; the
        // fallback is zero, so `first_non_zero..` is always in bounds.
        let top_bytes = unsafe { top_be.get_unchecked(first_non_zero..) };
        let total_bytes = top_bytes
            .len()
            .wrapping_add(lower_limbs.len().wrapping_mul(LIMB_BYTES));
        let mut bytes: Vec<u8> = Vec::with_capacity(total_bytes);

        bytes.extend_from_slice(top_bytes);
        for &limb in lower_limbs.iter().rev() {
            let limb_bytes: [u8; LIMB_BYTES] = limb.to_be_bytes();
            bytes.extend_from_slice(&limb_bytes);
        }
        bytes
    }

    /// Constructs an `InternalArbiUint` from a little-endian byte slice.
    ///
    /// The bytes are interpreted as an unsigned integer in little-endian order
    /// (least significant byte first). An empty slice is treated as zero.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }

        let num_limbs = bytes.len().div_ceil(LIMB_BYTES);
        let mut limbs: Vec<Limb> = Vec::with_capacity(num_limbs);

        for chunk in bytes.chunks(LIMB_BYTES) {
            let mut arr = [0_u8; LIMB_BYTES];
            // SAFETY: `chunks(LIMB_BYTES)` yields slices whose length is at
            // most `LIMB_BYTES`, exactly the destination array length.
            unsafe { arr.get_unchecked_mut(..chunk.len()) }.copy_from_slice(chunk);
            limbs.push(Limb::from_le_bytes(arr));
        }

        // from_limbs already strips trailing zero limbs — no manual pop loop needed.
        Self::from_limbs(limbs)
    }

    /// Constructs an `InternalArbiUint` from a big-endian byte slice.
    ///
    /// The bytes are interpreted as an unsigned integer in big-endian order
    /// (most significant byte first). An empty slice is treated as zero.
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }

        let num_limbs = bytes.len().div_ceil(LIMB_BYTES);
        let mut limbs: Vec<Limb> = Vec::with_capacity(num_limbs);

        // Process bytes from right to left (LSB first) in LIMB_BYTES-sized
        // chunks, converting each to an LE limb directly. This avoids the
        // intermediate reversed byte Vec that to_vec()+reverse() would need.
        let mut pos = bytes.len();
        while pos > 0 {
            let start = pos.saturating_sub(LIMB_BYTES);
            // SAFETY: start = pos.saturating_sub(LIMB_BYTES), so start <= pos <= bytes.len()
            let chunk = unsafe { bytes.get_unchecked(start..pos) };
            // chunk is in BE order; reverse within the limb for LE.
            let mut arr = [0_u8; LIMB_BYTES];
            for (j, &b) in chunk.iter().rev().enumerate() {
                // SAFETY: j < chunk.len() <= LIMB_BYTES
                unsafe {
                    *arr.get_unchecked_mut(j) = b;
                }
            }
            limbs.push(Limb::from_le_bytes(arr));
            pos = start;
        }

        Self::from_limbs(limbs)
    }
}

#[cfg(test)]
#[path = "tests/bytes.rs"]
mod tests;
