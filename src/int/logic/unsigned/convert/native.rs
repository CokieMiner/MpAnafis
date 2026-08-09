//! Native primitive conversions for the unsigned integer engine.

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

impl InternalMpUint {
    /// Constructs a value from one native limb.
    #[must_use]
    pub const fn from_limb(value: Limb) -> Self {
        if value == 0 {
            return Self::zero();
        }
        Self {
            repr: UintRepr::Inline {
                len: 1,
                limbs: [value, 0, 0, 0],
            },
        }
    }

    /// Creates an `InternalMpUint` from a `u64`.
    ///
    /// Returns `zero()` when `value` is zero, otherwise an `Inline` representation.
    /// Constructs limbs directly from the value without allocating a `Vec`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "extracting the low LIMB_BITS of a shifted primitive into one Limb"
    )]
    pub fn from_u64(value: u64) -> Self {
        const U64_LIMBS: usize = 64_usize.div_ceil(LIMB_BITS);
        if value == 0 {
            return Self::zero();
        }
        // U64_LIMBS is 1 on 64-bit, 2 on 32-bit, and 4 on 16-bit.
        // All cases fit within INLINE_LIMBS (4), so we build Inline directly.
        let mut arr = [0_usize; INLINE_LIMBS];
        let mut active_len: u8 = 0;
        for (i, slot) in arr.iter_mut().enumerate().take(U64_LIMBS) {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Index fits in u32"
            )]
            let shift = i.wrapping_mul(LIMB_BITS) as u32;
            let limb = value.wrapping_shr(shift) as Limb;
            *slot = limb;
            if limb != 0 {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "i < U64_LIMBS <= INLINE_LIMBS fits in u8"
                )]
                {
                    active_len = (i as u8).wrapping_add(1);
                }
            }
        }
        Self {
            repr: UintRepr::Inline {
                len: active_len,
                limbs: arr,
            },
        }
    }

    /// Creates an `InternalMpUint` from a `u128`.
    ///
    /// Returns `zero()` when `value` is zero.
    /// Constructs limbs directly from the value without allocating a `Vec`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "extracting the low LIMB_BITS of a shifted primitive into one Limb"
    )]
    pub fn from_u128(value: u128) -> Self {
        const U128_LIMBS: usize = 128_usize.div_ceil(LIMB_BITS);
        if value == 0 {
            return Self::zero();
        }
        // U128_LIMBS is 2 on 64-bit, 4 on 32-bit, and 8 on 16-bit.
        // On 64-bit and 32-bit, U128_LIMBS <= INLINE_LIMBS (4), so Inline.
        // On 16-bit, U128_LIMBS = 8 > INLINE_LIMBS, fall back to from_limbs.
        if U128_LIMBS <= INLINE_LIMBS {
            let mut arr = [0_usize; INLINE_LIMBS];
            let mut active_len: u8 = 0;
            for (i, slot) in arr.iter_mut().enumerate().take(U128_LIMBS) {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Index fits in u32"
                )]
                let shift = i.wrapping_mul(LIMB_BITS) as u32;
                let limb = value.wrapping_shr(shift) as Limb;
                *slot = limb;
                if limb != 0 {
                    #[allow(
                        clippy::as_conversions,
                        clippy::cast_possible_truncation,
                        reason = "i < U128_LIMBS <= INLINE_LIMBS fits in u8"
                    )]
                    {
                        active_len = (i as u8).wrapping_add(1);
                    }
                }
            }
            Self {
                repr: UintRepr::Inline {
                    len: active_len,
                    limbs: arr,
                },
            }
        } else {
            let mut limbs = [0_usize; U128_LIMBS];
            for (i, limb) in limbs.iter_mut().enumerate() {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Index fits in u32"
                )]
                let shift = i.wrapping_mul(LIMB_BITS) as u32;
                *limb = value.wrapping_shr(shift) as Limb;
            }
            Self::from_limbs(limbs.into())
        }
    }

    /// Attempts to convert to `u64`.
    ///
    /// Returns `None` if the value exceeds `u64::MAX`.
    /// Reads limbs directly without allocating a `Vec`.
    #[must_use]
    #[allow(
        unsafe_code,
        reason = "get_unchecked is safe: bounds are verified via limbs.len() == U64_LIMBS > 0"
    )]
    pub fn to_u64(&self) -> Option<u64> {
        const U64_LIMBS: usize = 64_usize.div_ceil(LIMB_BITS);
        let limbs = self.limbs();
        if limbs.is_empty() {
            return Some(0);
        }
        if limbs.len() > U64_LIMBS {
            return None;
        }
        // Check that the top limb does not exceed the u64 range.
        // When 64 is not a multiple of LIMB_BITS, the high limb must fit
        // within the remaining top bits.
        if limbs.len() == U64_LIMBS {
            #[allow(clippy::as_conversions, reason = "64 % LIMB_BITS fits in u32")]
            let top_bits = 64_usize.wrapping_rem(LIMB_BITS);
            if top_bits > 0 {
                // SAFETY: limbs.len() == U64_LIMBS > 0, so last index is valid.
                let top = unsafe { limbs.get_unchecked(limbs.len().wrapping_sub(1)) };
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Mask mathematically fits in Limb"
                )]
                let max_val = (1_u64.wrapping_shl(top_bits as u32).wrapping_sub(1)) as Limb;
                if *top > max_val {
                    return None;
                }
            }
        }
        let mut result: u64 = 0;
        for (i, &limb) in limbs.iter().enumerate() {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Index fits in u32"
            )]
            let shift = i.wrapping_mul(LIMB_BITS) as u32;
            #[allow(clippy::as_conversions, reason = "Limb always fits in u64/u128")]
            let shifted = (limb as u64).wrapping_shl(shift);
            result |= shifted;
        }
        Some(result)
    }

    /// Attempts to convert to `u128`.
    ///
    /// Returns `None` if the value exceeds `u128::MAX`.
    /// Reads limbs directly without allocating a `Vec`.
    #[must_use]
    #[allow(
        unsafe_code,
        reason = "get_unchecked is safe: bounds are verified via limbs.len() == U128_LIMBS > 0"
    )]
    pub fn to_u128(&self) -> Option<u128> {
        const U128_LIMBS: usize = 128_usize.div_ceil(LIMB_BITS);
        let limbs = self.limbs();
        if limbs.is_empty() {
            return Some(0);
        }
        if limbs.len() > U128_LIMBS {
            return None;
        }
        // Check that the top limb does not exceed the u128 range.
        if limbs.len() == U128_LIMBS {
            #[allow(clippy::as_conversions, reason = "128 % LIMB_BITS fits in u32")]
            let top_bits = 128_usize.wrapping_rem(LIMB_BITS);
            if top_bits > 0 {
                // SAFETY: limbs.len() == U128_LIMBS > 0, so last index is valid.
                let top = unsafe { limbs.get_unchecked(limbs.len().wrapping_sub(1)) };
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Mask mathematically fits in Limb"
                )]
                let max_val = (1_u128.wrapping_shl(top_bits as u32).wrapping_sub(1)) as Limb;
                if *top > max_val {
                    return None;
                }
            }
        }
        let mut result: u128 = 0;
        for (i, &limb) in limbs.iter().enumerate() {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Index fits in u32"
            )]
            let shift = i.wrapping_mul(LIMB_BITS) as u32;
            #[allow(clippy::as_conversions, reason = "Limb always fits in u64/u128")]
            let shifted = (limb as u128).wrapping_shl(shift);
            result |= shifted;
        }
        Some(result)
    }

    /// Attempts to convert to `usize`.
    ///
    /// Returns `None` if the value exceeds `usize::MAX`.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        let limbs = self.limbs();
        if limbs.is_empty() {
            return Some(0);
        }
        if limbs.len() == 1 {
            return limbs.first().copied();
        }
        None
    }
}
