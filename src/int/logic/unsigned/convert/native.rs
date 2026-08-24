//! Native primitive conversions for the unsigned integer engine.

#[cfg(target_pointer_width = "16")]
use super::LIMB_BITS;
use super::{InternalMpUint, Limb};

impl InternalMpUint {
    /// Creates an `InternalMpUint` from a `u64`.
    ///
    /// Returns `zero()` when `value` is zero, otherwise an `Inline` representation.
    /// Constructs limbs directly from the value without allocating a `Vec`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "extracting limbs of native primitive types"
    )]
    pub const fn from_u64(value: u64) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self::from_limb(value as Limb)
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self::from_limbs_2(value as Limb, (value >> 32) as Limb)
        }
        #[cfg(target_pointer_width = "16")]
        {
            Self::from_limbs_4(
                value as Limb,
                (value >> 16) as Limb,
                (value >> 32) as Limb,
                (value >> 48) as Limb,
            )
        }
    }

    /// Creates an `InternalMpUint` from a `u128`.
    ///
    /// Returns `zero()` when `value` is zero.
    /// Constructs limbs directly from the value without allocating a `Vec`.
    #[must_use]
    #[cfg(not(target_pointer_width = "16"))]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "extracting limbs of native primitive types"
    )]
    pub const fn from_u128(value: u128) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self::from_limbs_2(value as Limb, (value >> 64) as Limb)
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self::from_limbs_4(
                value as Limb,
                (value >> 32) as Limb,
                (value >> 64) as Limb,
                (value >> 96) as Limb,
            )
        }
    }

    /// Creates an `InternalMpUint` from a `u128`.
    ///
    /// Returns `zero()` when `value` is zero.
    /// Constructs limbs directly from the value without allocating a `Vec`.
    #[must_use]
    #[cfg(target_pointer_width = "16")]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "extracting limbs of native primitive types"
    )]
    pub fn from_u128(value: u128) -> Self {
        const U128_LIMBS: usize = 8;
        let mut limbs = [0_usize; U128_LIMBS];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let shift = i.wrapping_mul(LIMB_BITS) as u32;
            *limb = value.wrapping_shr(shift) as Limb;
        }
        Self::from_limbs(limbs.into())
    }

    /// Attempts to convert to `u64`.
    ///
    /// Returns `None` if the value exceeds `u64::MAX`.
    /// Reads limbs directly without allocating a `Vec`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        reason = "Limb fits in u64 on 16/32/64-bit platforms"
    )]
    pub fn to_u64(&self) -> Option<u64> {
        #[cfg(target_pointer_width = "64")]
        {
            if self.limbs().len() > 1 {
                return None;
            }
            let [lo, ..] = self.extract_4();
            Some(lo as u64)
        }
        #[cfg(target_pointer_width = "32")]
        {
            if self.limbs().len() > 2 {
                return None;
            }
            let [lo, hi, ..] = self.extract_4();
            Some(((hi as u64) << 32) | (lo as u64))
        }
        #[cfg(target_pointer_width = "16")]
        {
            if self.limbs().len() > 4 {
                return None;
            }
            let [l0, l1, l2, l3] = self.extract_4();
            Some(((l3 as u64) << 48) | ((l2 as u64) << 32) | ((l1 as u64) << 16) | (l0 as u64))
        }
    }

    /// Attempts to convert to `u128`.
    ///
    /// Returns `None` if the value exceeds `u128::MAX`.
    /// Reads limbs directly without allocating a `Vec`.
    #[must_use]
    #[allow(
        clippy::as_conversions,
        reason = "Limb fits in u128 on 16/32/64-bit platforms"
    )]
    pub fn to_u128(&self) -> Option<u128> {
        #[cfg(target_pointer_width = "64")]
        {
            if self.limbs().len() > 2 {
                return None;
            }
            let [lo, hi, ..] = self.extract_4();
            Some(((hi as u128) << 64) | (lo as u128))
        }
        #[cfg(target_pointer_width = "32")]
        {
            if self.limbs().len() > 4 {
                return None;
            }
            let [l0, l1, l2, l3] = self.extract_4();
            Some(((l3 as u128) << 96) | ((l2 as u128) << 64) | ((l1 as u128) << 32) | (l0 as u128))
        }
        #[cfg(target_pointer_width = "16")]
        {
            if self.limbs().len() > 8 {
                return None;
            }
            let mut result: u128 = 0;
            for (i, &limb) in self.limbs().iter().enumerate() {
                let shift = i.wrapping_mul(LIMB_BITS) as u32;
                let shifted = (limb as u128).wrapping_shl(shift);
                result |= shifted;
            }
            Some(result)
        }
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
