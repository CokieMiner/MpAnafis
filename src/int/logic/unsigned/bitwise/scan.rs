//! Unsigned bit scanning and bit-count queries.

#![allow(
    unsafe_code,
    reason = "Bypassing bounds checks on loop-bounded indices where the range guarantees in-bounds access."
)]

use core::cmp::min;

use super::{InternalArbiUint, LIMB_BITS, Limb};

impl InternalArbiUint {
    /// Returns the number of ones in the binary representation.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.limbs()
            .iter()
            .map(|&limb| {
                #[allow(clippy::as_conversions, reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks.")]
                let count = limb.count_ones() as usize;
                count
            })
            .sum()
    }

    /// Returns the number of trailing zero bits.
    ///
    /// Returns `0` when the value is zero.
    #[inline]
    #[must_use]
    pub fn trailing_zeros(&self) -> usize {
        let mut count: usize = 0;
        for &limb in self.limbs() {
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
            )]
            let tz = limb.trailing_zeros() as usize;
            count = count.wrapping_add(tz);
            // trailing_zeros returns LIMB_BITS for zero; any other value means
            // we found the first set bit and can return immediately.
            if tz != LIMB_BITS {
                return count;
            }
        }
        // Value is zero
        0
    }

    /// Returns `true` if any bit at a position strictly less than `bits` is set.
    #[inline]
    #[must_use]
    pub fn has_any_bits_set_below(&self, bits: usize) -> bool {
        if bits == 0 {
            return false;
        }
        let full_limbs = bits.wrapping_div(LIMB_BITS);
        let rem_bits = bits.wrapping_rem(LIMB_BITS);
        let limbs = self.limbs();
        let min_full = min(full_limbs, limbs.len());
        if limbs.iter().take(min_full).any(|&limb| limb != 0) {
            return true;
        }
        if rem_bits != 0
            && let Some(&limb) = limbs.get(min_full)
        {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS difference fits in u32: using 'as' avoids checked conversions."
            )]
            let mask = if rem_bits == LIMB_BITS {
                Limb::MAX
            } else {
                Limb::MAX.wrapping_shr(LIMB_BITS.wrapping_sub(rem_bits) as u32)
            };
            return (limb & mask) != 0;
        }
        false
    }

    /// Finds the position of the first (least-significant) zero bit.
    ///
    /// Because an arbitrary-precision integer has conceptually infinite
    /// leading zeros, this always returns `Some` — even for zero, where
    /// bit `0` is the first zero bit.
    #[must_use]
    pub fn find_first_zero_bit(&self) -> usize {
        let mut pos: usize = 0;
        for &limb in self.limbs() {
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
            )]
            let trailing_ones = limb.trailing_ones() as usize;
            // trailing_ones returns LIMB_BITS for !0 (all ones); any other
            // value means we found the first zero bit.
            pos = pos.wrapping_add(trailing_ones);
            if trailing_ones != LIMB_BITS {
                return pos;
            }
        }
        // No limbs or all limbs are all-ones; for empty (zero) the first
        // zero bit is at position 0; for an all-ones value the first zero
        // bit is at the current position (conceptually beyond all limbs,
        // which is correct for unlimited precision).
        pos
    }

    /// Returns the index of the least significant set bit, or `None` if zero.
    #[must_use]
    pub fn find_first_set_bit(&self) -> Option<usize> {
        let mut pos: usize = 0;
        for &limb in self.limbs() {
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
            )]
            let trailing_zeros = limb.trailing_zeros() as usize;
            // trailing_zeros returns LIMB_BITS for zero; any other value
            // means we found the first set bit.
            pos = pos.wrapping_add(trailing_zeros);
            if trailing_zeros != LIMB_BITS {
                return Some(pos);
            }
        }
        None
    }

    /// Returns the number of significant bits in the binary representation.
    /// Returns 0 if the value is zero.
    #[must_use]
    pub fn significant_bits(&self) -> usize {
        let limbs = self.limbs();
        if limbs.is_empty() {
            return 0;
        }
        let last_idx = limbs.len().wrapping_sub(1);
        // SAFETY: last_idx = limbs.len() - 1 is valid because is_empty() returned false.
        let last_limb = unsafe { *limbs.get_unchecked(last_idx) };
        if last_limb == 0 {
            // Should not happen for normalized numbers, but just in case
            return 0;
        }
        #[allow(
            clippy::as_conversions,
            reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
        )]
        let lz = last_limb.leading_zeros() as usize;
        let bits_in_last = LIMB_BITS.wrapping_sub(lz);
        last_idx.wrapping_mul(LIMB_BITS).wrapping_add(bits_in_last)
    }

    /// Returns the required bit width for bounded storage of this magnitude.
    #[must_use]
    pub fn required_unsigned_bits_for_bounded_storage(&self) -> usize {
        if self.is_zero() {
            1
        } else {
            self.significant_bits()
        }
    }

    /// Returns the number of leading zeros relative to the given width.
    #[must_use]
    pub fn leading_zeros_for_width(&self, width: usize) -> usize {
        if self.is_zero() {
            return width;
        }
        let sig = self.significant_bits();
        if sig >= width {
            0
        } else {
            width.wrapping_sub(sig)
        }
    }

    /// Returns the number of leading ones within the given width.
    #[must_use]
    pub fn leading_ones_for_width(&self, width: usize) -> usize {
        if width == 0 || self.is_zero() {
            return 0;
        }
        let limbs = self.limbs();
        let mut count: usize = 0;
        let mut remaining = width;

        let top_limb_idx = (width.wrapping_sub(1)).wrapping_div(LIMB_BITS);
        let top_bits = (width.wrapping_sub(1))
            .wrapping_rem(LIMB_BITS)
            .wrapping_add(1);

        for limb_idx in (0..=top_limb_idx).rev() {
            let limb = limbs.get(limb_idx).copied().unwrap_or(0);
            let bits_here = if limb_idx == top_limb_idx {
                top_bits
            } else {
                min(remaining, LIMB_BITS)
            };
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "bits_here <= LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
            )]
            let bits_here_u32 = bits_here as u32;

            if bits_here == LIMB_BITS {
                if limb == !0 {
                    count = count.wrapping_add(LIMB_BITS);
                } else {
                    #[allow(
                        clippy::as_conversions,
                        reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
                    )]
                    let ones = limb.leading_ones() as usize;
                    count = count.wrapping_add(ones);
                    break;
                }
            } else {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
                )]
                let limb_bits_u32 = LIMB_BITS as u32;
                let shift = limb_bits_u32.wrapping_sub(bits_here_u32);
                let window = limb & ((1_usize).wrapping_shl(bits_here_u32)).wrapping_sub(1);
                let aligned = window.wrapping_shl(shift);
                #[allow(
                    clippy::as_conversions,
                    reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
                )]
                let ones = aligned.leading_ones() as usize;
                count = count.wrapping_add(ones);
                if ones < bits_here {
                    break;
                }
            }

            remaining = remaining.wrapping_sub(bits_here);
            if remaining == 0 {
                break;
            }
        }

        count
    }

    /// Counts the number of trailing (least-significant) one bits.
    #[must_use]
    pub fn trailing_ones(&self) -> usize {
        let mut count: usize = 0;
        for &limb in self.limbs() {
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize, avoiding branchy checks."
            )]
            let ones = limb.trailing_ones() as usize;
            // trailing_ones returns LIMB_BITS for !0 (all ones); any other
            // value means we found the first zero bit.
            count = count.wrapping_add(ones);
            if ones != LIMB_BITS {
                return count;
            }
        }
        count
    }

    /// Returns the number of zero bits within the given width.
    #[must_use]
    pub fn count_zeros_for_width(&self, width: usize) -> usize {
        if width == 0 {
            return 0;
        }
        let full_limbs = width.wrapping_div(LIMB_BITS);
        let rem_bits = width.wrapping_rem(LIMB_BITS);
        let limbs = self.limbs();
        let mut ones = 0_usize;

        for &limb in limbs.iter().take(full_limbs) {
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize."
            )]
            {
                ones = ones.wrapping_add(limb.count_ones() as usize);
            }
        }

        if rem_bits != 0
            && let Some(&limb) = limbs.get(full_limbs)
        {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "rem_bits < LIMB_BITS <= 64 fits in u32"
            )]
            let rem_bits_u32 = rem_bits as u32;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS is 16, 32, or 64 and always fits in u32"
            )]
            let limb_bits_u32 = LIMB_BITS as u32;
            let mask = Limb::MAX.wrapping_shr(limb_bits_u32.wrapping_sub(rem_bits_u32));
            #[allow(
                clippy::as_conversions,
                reason = "u32 fits in usize: since the maximum count is LIMB_BITS (<= 64), it always fits in a 16-bit or wider usize."
            )]
            {
                ones = ones.wrapping_add((limb & mask).count_ones() as usize);
            }
        }

        width.wrapping_sub(ones)
    }
}

#[cfg(test)]
#[path = "tests/scan.rs"]
mod tests;
