//! Primitive property queries and floating-point conversion helpers.

use core::cmp::min;

use alloc::vec;

use super::{InternalMpUint, LIMB_BITS, Limb};
impl InternalMpUint {
    /// Returns true if the integer is exactly zero.
    /// Because `InternalMpUint` strictly normalizes zero to an empty vector,
    /// this check is an extremely fast `O(1)` length comparison.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs().is_empty()
    }

    /// Returns true if the integer is exactly one.
    #[inline]
    #[must_use]
    pub fn is_one(&self) -> bool {
        let l = self.limbs();
        l.len() == 1 && l.first() == Some(&1)
    }

    /// Returns true if the integer is even.
    #[inline]
    #[must_use]
    pub fn is_even(&self) -> bool {
        // Zero has an empty limb vector, and first().copied().unwrap_or(0)
        // yields 0, so 0 & 1 == 0 is true — the explicit is_zero() guard
        // is redundant.
        (self.limbs().first().copied().unwrap_or(0) & 1) == 0
    }

    /// Returns true if the integer is odd.
    #[inline]
    #[must_use]
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    /// Returns `true` when `self` is a power of two.
    ///
    /// Uses an early-exit scan instead of `count_ones()` to avoid traversing
    /// all limbs when the answer is already known.
    #[inline]
    #[must_use]
    pub fn is_power_of_two(&self) -> bool {
        let Some((last, rest)) = self.limbs().split_last() else {
            return false;
        };
        if !last.is_power_of_two() {
            return false;
        }
        rest.iter().all(|&limb| limb == 0)
    }

    /// Returns whether shifting this value left exceeds a caller-proved
    /// bounded unsigned width.
    #[inline]
    #[must_use]
    pub fn bounded_shl_overflows(&self, bits: usize, shift: usize) -> bool {
        let value_bits = self.significant_bits();
        debug_assert!(
            value_bits <= bits,
            "bounded unsigned value must fit its declared precision"
        );
        // For non-zero x, unsigned storage requires `value_bits + shift` bits.
        // The caller proves `value_bits <= bits`, so the wrapping subtraction
        // equals the exact non-negative slack and avoids a second checked path.
        value_bits != 0 && shift > bits.wrapping_sub(value_bits)
    }

    /// Returns the value `2^bits` (a power of two with a single bit set).
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo LIMB_BITS fits in u32"
    )]
    pub fn power_of_two(bits: usize) -> Self {
        let limb_idx = bits.wrapping_div(LIMB_BITS);
        let bit_in_limb = (bits.wrapping_rem(LIMB_BITS)) as u32;
        let mut limbs = alloc::vec![0; limb_idx.wrapping_add(1)];
        if let Some(limb) = limbs.get_mut(limb_idx) {
            *limb = 1_usize.wrapping_shl(bit_in_limb);
        }
        Self::from_limbs(limbs)
    }

    /// Returns the maximum value representable with `bits` bits.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo Limb::BITS fits in u32"
    )]
    pub fn max_for_bits(bits: usize) -> Self {
        if bits == 0 {
            return Self::zero();
        }
        let num_limbs = bits.div_ceil(LIMB_BITS);
        let mut limbs = vec![Limb::MAX; num_limbs];
        let rem = bits.wrapping_rem(LIMB_BITS);
        if rem != 0 {
            let mask = (1_usize.wrapping_shl(rem as u32)).wrapping_sub(1);
            if let Some(last) = limbs.last_mut() {
                *last = mask;
            }
        }
        Self::from_limbs(limbs)
    }

    /// Converts the value to `f64`, returning `None` if the value is too large
    /// (f64 overflow).
    ///
    /// Computes `significant_bits` inline from the top limb to avoid a separate
    /// limb traversal.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        let limbs = self.limbs();
        let n = limbs.len();
        if n == 0 {
            return Some(0.0);
        }
        // Compute significant_bits inline: fetch top limb and count leading zeros.
        #[allow(unsafe_code, reason = "n >= 1 is verified by the empty check above")]
        // SAFETY: n >= 1 is verified by the empty check above
        let hi = unsafe { *limbs.get_unchecked(n.wrapping_sub(1)) };
        #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
        let lz = hi.leading_zeros() as usize;
        let bits = n
            .wrapping_sub(1)
            .wrapping_mul(LIMB_BITS)
            .wrapping_add(LIMB_BITS.wrapping_sub(lz));
        // f64 max exponent is 1023. So max value is ~2^1024
        if bits > 1024 {
            return None;
        }

        let mantissa = leading_bits_as_u64(limbs, bits, 53);

        let exponent = bits.wrapping_sub(1).wrapping_add(1023);
        #[allow(
            clippy::as_conversions,
            reason = "mantissa always ≤ 53 bits, exponent 0..=2047 — both always fit in u64"
        )]
        {
            let mantissa_bits = mantissa & 0x000F_FFFF_FFFF_FFFF;
            let bits_u64 = (exponent as u64).wrapping_shl(52) | mantissa_bits;
            Some(f64::from_bits(bits_u64))
        }
    }

    /// Converts the value to `f32`, returning `None` if the value is too large
    /// (f32 overflow).
    ///
    /// Computes `significant_bits` inline from the top limb to avoid a separate
    /// limb traversal.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        let limbs = self.limbs();
        let n = limbs.len();
        if n == 0 {
            return Some(0.0);
        }
        // Compute significant_bits inline: fetch top limb and count leading zeros.
        #[allow(unsafe_code, reason = "n >= 1 is verified by the empty check above")]
        // SAFETY: n >= 1 is verified by the empty check above
        let hi = unsafe { *limbs.get_unchecked(n.wrapping_sub(1)) };
        #[allow(clippy::as_conversions, reason = "u32 fits in usize")]
        let lz = hi.leading_zeros() as usize;
        let bits = n
            .wrapping_sub(1)
            .wrapping_mul(LIMB_BITS)
            .wrapping_add(LIMB_BITS.wrapping_sub(lz));
        // f32 max exponent is 127. So max value is ~2^128
        if bits > 128 {
            return None;
        }

        let mantissa = leading_bits_as_u64(limbs, bits, 24);

        let exponent = bits.wrapping_sub(1).wrapping_add(127);
        let mantissa_bits = mantissa & 0x007F_FFFF;
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "exponent <= 254 (bits checked <= 128) and mantissa_bits masked to 23 bits — both always fit in u32"
        )]
        let bits_u32 = (exponent as u32).wrapping_shl(23) | (mantissa_bits as u32);
        Some(f32::from_bits(bits_u32))
    }
}

#[inline]
#[allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "bit index modulo LIMB_BITS always fits in u32; get_unchecked safe by construction"
)]
fn leading_bits_as_u64(limbs: &[Limb], significant_bits: usize, width: usize) -> u64 {
    let take = min(significant_bits, width);
    if take == 0 {
        return 0;
    }

    let mut acc: u64 = 0;
    let mut remaining = take;
    let mut bit_pos = significant_bits;

    while remaining > 0 {
        let top_bit = bit_pos.wrapping_sub(1);
        let limb_idx = top_bit.wrapping_div(LIMB_BITS);
        let bit_in_limb = top_bit.wrapping_rem(LIMB_BITS);
        // SAFETY: top_bit < significant_bits, and significant_bits is derived
        // from the normalized top limb, so limb_idx is within limbs.
        let limb = unsafe { *limbs.get_unchecked(limb_idx) };
        let available = bit_in_limb.wrapping_add(1);
        let take_now = min(remaining, available);
        let low_bit = available.wrapping_sub(take_now);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "take_now <= width <= 53 for f64/f32 callers, and low_bit < LIMB_BITS <= 64"
        )]
        let take_now_u32 = take_now as u32;
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "low_bit < LIMB_BITS <= 64 fits in u32"
        )]
        let low_bit_u32 = low_bit as u32;
        let mask = if take_now == 64 {
            u64::MAX
        } else {
            1_u64.wrapping_shl(take_now_u32).wrapping_sub(1)
        };
        let bits = (limb as u64).wrapping_shr(low_bit_u32) & mask;
        // The loop walks the requested window from most-significant chunk to
        // least-significant chunk. Therefore appending each chunk by shifting
        // the accumulator left by its width preserves numeric bit order.
        acc = acc.wrapping_shl(take_now_u32) | bits;
        remaining = remaining.wrapping_sub(take_now);
        bit_pos = bit_pos.wrapping_sub(take_now);
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "pad fits in u32"
    )]
    acc.wrapping_shl((width.wrapping_sub(take)) as u32)
}

#[cfg(test)]
#[path = "tests/properties.rs"]
mod tests;
