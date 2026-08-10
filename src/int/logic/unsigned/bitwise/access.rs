//! Individual-bit access, mutation, range extraction, and forward searches.

#![allow(
    unsafe_code,
    reason = "Bypassing bounds checks on loop-bounded indices where the range guarantees in-bounds access."
)]

use alloc::vec::Vec;

use super::{InternalArbiUint, LIMB_BITS, Limb};

impl InternalArbiUint {
    /// Returns the bit at position `bit` (0-indexed, LSB first).
    #[must_use]
    pub fn get_bit(&self, bit: usize) -> bool {
        let limb_idx = bit.wrapping_div(LIMB_BITS);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "modulo LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
        )]
        let bit_in_limb = bit.wrapping_rem(LIMB_BITS) as u32;
        self.limbs()
            .get(limb_idx)
            .is_some_and(|&limb| (limb >> bit_in_limb) & 1 == 1)
    }

    /// Sets the bit at position `bit` to `value`.
    #[must_use]
    pub fn set_bit_to(&self, bit: usize, value: bool) -> Self {
        if self.get_bit(bit) == value {
            return self.clone();
        }
        let limb_idx = bit.wrapping_div(LIMB_BITS);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "modulo LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
        )]
        let bit_in_limb = bit.wrapping_rem(LIMB_BITS) as u32;
        let mut result = self.clone();
        if result.limbs().len() <= limb_idx {
            result.resize(limb_idx.wrapping_add(1));
        }
        // SAFETY: limb_idx < result.limbs().len() by construction after resize.
        unsafe {
            let dst = result.limbs_mut().as_mut_ptr().add(limb_idx);
            if value {
                *dst |= (1_usize).wrapping_shl(bit_in_limb);
            } else {
                *dst &= !((1_usize).wrapping_shl(bit_in_limb));
            }
        }
        result.normalize();
        result
    }

    /// Extracts bits `[start..end)` (exclusive end) as a new integer.
    #[must_use]
    pub fn bit_range(&self, start: usize, end: usize) -> Self {
        if start >= end {
            return Self::zero();
        }
        let limbs = self.limbs();
        let start_limb = start.wrapping_div(LIMB_BITS);
        if start_limb >= limbs.len() {
            return Self::zero();
        }
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "modulo LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
        )]
        let bit_offset = start.wrapping_rem(LIMB_BITS) as u32;
        let width_bits = end.wrapping_sub(start);
        let result_len = width_bits
            .wrapping_add(LIMB_BITS.wrapping_sub(1))
            .wrapping_div(LIMB_BITS);
        let mut result = Vec::with_capacity(result_len);

        for index in 0..result_len {
            let src_idx = start_limb.wrapping_add(index);
            let low = limbs.get(src_idx).copied().unwrap_or(0);
            let value = if bit_offset == 0 {
                low
            } else {
                let high = limbs.get(src_idx.wrapping_add(1)).copied().unwrap_or(0);
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "LIMB_BITS fits in u32: using 'as' avoids checked conversions."
                )]
                let shift_up = (LIMB_BITS as u32).wrapping_sub(bit_offset);
                (low >> bit_offset) | (high << shift_up)
            };
            result.push(value);
        }

        let remaining = width_bits.wrapping_rem(LIMB_BITS);
        if remaining != 0
            && let Some(last) = result.last_mut()
        {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS difference fits in u32: using 'as' avoids checked conversions."
            )]
            let mask = Limb::MAX.wrapping_shr(LIMB_BITS.wrapping_sub(remaining) as u32);
            *last &= mask;
        }

        Self::from_limbs(result)
    }

    /// Finds the next set bit at or after position `from`.
    #[must_use]
    pub fn find_next_set_bit(&self, from: usize) -> Option<usize> {
        let start_limb = from.wrapping_div(LIMB_BITS);
        let limbs = self.limbs();
        if start_limb >= limbs.len() {
            return None;
        }
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "modulo LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
        )]
        let start_bit = from.wrapping_rem(LIMB_BITS) as u32;

        // SAFETY: start_limb < limbs.len(), checked above.
        let first_limb = unsafe { *limbs.get_unchecked(start_limb) };
        let first_mask = !(((1_usize).wrapping_shl(start_bit)).wrapping_sub(1));
        let masked = first_limb & first_mask;
        #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
        let trailing_zeros = masked.trailing_zeros() as usize;
        if trailing_zeros != LIMB_BITS {
            return Some(
                start_limb
                    .wrapping_mul(LIMB_BITS)
                    .wrapping_add(trailing_zeros),
            );
        }

        for index in start_limb.wrapping_add(1)..limbs.len() {
            // SAFETY: index is in start_limb + 1 .. limbs.len().
            let limb = unsafe { *limbs.get_unchecked(index) };
            #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
            let next_trailing_zeros = limb.trailing_zeros() as usize;
            if next_trailing_zeros != LIMB_BITS {
                return Some(
                    index
                        .wrapping_mul(LIMB_BITS)
                        .wrapping_add(next_trailing_zeros),
                );
            }
        }
        None
    }

    /// Finds the next zero bit at or after position `from`.
    ///
    /// Because unlimited integers have infinite zero bits beyond the current
    /// storage, this always returns a value.
    #[must_use]
    pub fn find_next_zero_bit(&self, from: usize) -> usize {
        let start_limb = from.wrapping_div(LIMB_BITS);
        let limbs = self.limbs();

        if start_limb < limbs.len() {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "modulo LIMB_BITS fits in u32: using 'as' avoids checked conversions and results in branchless register truncation."
            )]
            let start_bit = from.wrapping_rem(LIMB_BITS) as u32;
            let first_mask = ((1_usize).wrapping_shl(start_bit)).wrapping_sub(1);

            // SAFETY: start_limb < limbs.len(), checked above.
            let first_limb = unsafe { *limbs.get_unchecked(start_limb) };
            let masked = first_limb | first_mask;
            #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
            let trailing_zeros = (!masked).trailing_zeros() as usize;
            if trailing_zeros != LIMB_BITS {
                return start_limb
                    .wrapping_mul(LIMB_BITS)
                    .wrapping_add(trailing_zeros);
            }

            for index in start_limb.wrapping_add(1)..limbs.len() {
                // SAFETY: index is in start_limb + 1 .. limbs.len().
                let limb = unsafe { *limbs.get_unchecked(index) };
                #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
                let next_trailing_zeros = (!limb).trailing_zeros() as usize;
                if next_trailing_zeros != LIMB_BITS {
                    return index
                        .wrapping_mul(LIMB_BITS)
                        .wrapping_add(next_trailing_zeros);
                }
            }
        }

        let beyond = limbs.len().wrapping_mul(LIMB_BITS);
        if from > beyond { from } else { beyond }
    }
}
