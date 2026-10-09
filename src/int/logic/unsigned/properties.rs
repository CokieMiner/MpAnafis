//! Scalar predicates and construction of binary precision bounds.

#![expect(
    unsafe_code,
    reason = "Binary bound constructors initialize every limb and prove their nonzero highest mask before normalized construction"
)]

use alloc::vec;

use super::{InternalMpUint, LIMB_BITS, Limb};

impl InternalMpUint {
    /// Returns whether the canonical magnitude has no limbs.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs().is_empty()
    }

    /// Returns whether the magnitude is one.
    #[inline]
    #[must_use]
    pub fn is_one(&self) -> bool {
        matches!(self.limbs(), [1])
    }

    /// Returns whether the low bit is zero, including canonical zero.
    #[inline]
    #[must_use]
    pub fn is_even(&self) -> bool {
        (self.limbs().first().copied().unwrap_or(0) & 1) == 0
    }

    /// Returns whether the low bit is one.
    #[inline]
    #[must_use]
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    /// Returns whether exactly one bit is set.
    #[inline]
    #[must_use]
    pub fn is_power_of_two(&self) -> bool {
        let Some((last, rest)) = self.limbs().split_last() else {
            return false;
        };
        last.is_power_of_two() && rest.iter().all(|&limb| limb == 0)
    }

    /// Tests whether a left shift exceeds `bits`, given `self < 2^bits`.
    #[inline]
    #[must_use]
    pub fn bounded_shl_overflows(&self, bits: usize, shift: usize) -> bool {
        let value_bits = self.significant_bits();
        debug_assert!(value_bits <= bits, "the magnitude fits its declared width");
        // SAFETY: bounded callers establish value_bits <= bits; their exact
        // nonnegative precision slack fits usize on every supported target.
        let slack = unsafe { bits.unchecked_sub(value_bits) };
        value_bits != 0 && shift > slack
    }

    /// Constructs `2^bits`.
    #[must_use]
    pub fn power_of_two(bits: usize) -> Self {
        let index = bits.div_euclid(LIMB_BITS);
        let value = 1_usize << bits.rem_euclid(LIMB_BITS);
        match index {
            0 => Self::from_limb(value),
            1 => Self::from_limbs_2(0, value),
            2 => Self::from_limbs_4(0, 0, value, 0),
            3 => Self::from_limbs_4(0, 0, 0, value),
            _ => {
                // SAFETY: index <= usize::MAX / LIMB_BITS and LIMB_BITS >= 16,
                // so index + 1 fits usize. The checked size cannot fail, and
                // infallible extraction excludes an unreachable allocation branch.
                let len = unsafe { index.checked_add(1).unwrap_unchecked() };
                let mut limbs = vec![0; len];
                // SAFETY: len = index + 1 bounds this initialized exclusive slot.
                unsafe {
                    *limbs.get_unchecked_mut(index) = value;
                }
                // SAFETY: all limbs are initialized and the last limb is nonzero.
                unsafe { Self::from_limbs_normalized(limbs) }
            }
        }
    }

    /// Constructs `2^bits - 1`, including zero for `bits == 0`.
    #[must_use]
    pub fn max_for_bits(bits: usize) -> Self {
        if bits == 0 {
            return Self::zero();
        }
        let len = bits.div_ceil(LIMB_BITS);
        let remainder = bits.rem_euclid(LIMB_BITS);
        let mask = if remainder == 0 {
            Limb::MAX
        } else {
            // SAFETY: 0 < remainder < LIMB_BITS bounds the positive difference
            // below the limb width on every target and makes the mask nonzero.
            Limb::MAX >> unsafe { LIMB_BITS.unchecked_sub(remainder) }
        };
        match len {
            1 => Self::from_limb(mask),
            2 => Self::from_limbs_2(Limb::MAX, mask),
            3 => Self::from_limbs_4(Limb::MAX, Limb::MAX, mask, 0),
            4 => Self::from_limbs_4(Limb::MAX, Limb::MAX, Limb::MAX, mask),
            _ => {
                let mut limbs = vec![Limb::MAX; len];
                // SAFETY: len > 4 proves the initialized vector is nonempty.
                *unsafe { limbs.last_mut().unwrap_unchecked() } = mask;
                // SAFETY: every limb is initialized and the top mask is nonzero.
                unsafe { Self::from_limbs_normalized(limbs) }
            }
        }
    }
}
