//! Wrapping operations for internal big integers.

use super::{InternalArbiUint, LIMB_BITS, UintRepr};

impl InternalArbiUint {
    /// Masks `self` to `bits` bits for wrapping semantics.
    ///
    /// When the value is heap-allocated, the internal `Vec<Limb>` is mutated in
    /// place to avoid an allocation and copy.
    #[inline]
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo LIMB_BITS fits in u32"
    )]
    pub fn apply_wrapping(self, bits: usize) -> Self {
        let sig = self.significant_bits();
        if sig <= bits {
            return self;
        }
        let keep = bits.wrapping_add(LIMB_BITS - 1).wrapping_div(LIMB_BITS);
        let rem = bits.wrapping_rem(LIMB_BITS);
        self.apply_wrapping_with_mask(keep, rem)
    }

    /// Compute `(2^bits - value) mod 2^bits` in-place without creating intermediate allocations.
    ///
    /// Used for bounded wrapping subtraction underflow when `a < b`, avoiding the creation
    /// of multiple temporary integers.
    #[inline]
    #[must_use]
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "modulo LIMB_BITS fits in u32"
    )]
    #[allow(
        unsafe_code,
        reason = "resizing and mutating slice in-place via raw slice extraction"
    )]
    pub fn apply_negate_wrapping(mut self, bits: usize) -> Self {
        if self.is_zero() || bits == 0 {
            return Self::zero();
        }
        let keep = bits.wrapping_add(LIMB_BITS - 1).wrapping_div(LIMB_BITS);
        let rem = bits.wrapping_rem(LIMB_BITS);
        let old_len = self.limbs().len();
        // SAFETY: ensure_capacity_set_len_get_limbs returns a valid slice of length keep.
        let limbs = unsafe { self.ensure_capacity_set_len_get_limbs(keep) };
        if old_len < keep {
            for limb in limbs.iter_mut().skip(old_len) {
                *limb = 0;
            }
        }
        let ptr = limbs.as_mut_ptr();
        let mut i = 0_usize;
        while i < keep {
            // SAFETY: i < keep, ptr is valid for keep elements.
            let limb = unsafe { *ptr.add(i) };
            if limb != 0 {
                // SAFETY: i < keep, ptr is valid.
                unsafe {
                    *ptr.add(i) = limb.wrapping_neg();
                }
                i = i.wrapping_add(1);
                break;
            }
            i = i.wrapping_add(1);
        }
        while i < keep {
            // SAFETY: i < keep, ptr is valid.
            unsafe {
                *ptr.add(i) = !*ptr.add(i);
            }
            i = i.wrapping_add(1);
        }
        if rem != 0 {
            let mask = (1_usize.wrapping_shl(rem as u32)).wrapping_sub(1);
            // SAFETY: keep >= 1, so keep - 1 is in bounds.
            unsafe {
                *ptr.add(keep.wrapping_sub(1)) &= mask;
            }
        }
        self.normalize();
        self
    }

    /// Mask an internal value to `bits` bits using precomputed `(keep, rem)`.
    ///
    /// `keep` is the number of limbs to retain (`ceil(bits / LIMB_BITS)`).
    /// `rem` is the bit remainder within the last limb (`bits % LIMB_BITS`).
    ///
    /// Callers that already have `keep` and `rem` available can use this to
    /// avoid recomputing the division and remainder.
    #[inline]
    #[must_use]
    #[allow(
        unsafe_code,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Modulo LIMB_BITS fits in u32; nonzero remainder proves the retained slice has a last limb."
    )]
    pub fn apply_wrapping_with_mask(mut self, keep: usize, rem: usize) -> Self {
        let current_len = self.limbs().len();
        // Fast path: no wrapping needed when keep > current limb count,
        // or keep == current limb count and rem == 0.
        if keep > current_len || (keep == current_len && rem == 0) {
            return self;
        }

        // When keep == current_len but rem != 0, we only need to mask the
        // last limb in place -- no truncation required.
        if keep == current_len {
            let mask = (1_usize.wrapping_shl(rem as u32)).wrapping_sub(1);
            // SAFETY: rem != 0 here implies keep >= 1, and keep == current_len.
            let last = unsafe { self.limbs_mut().last_mut().unwrap_unchecked() };
            *last &= mask;
            self.normalize();
            return self;
        }

        // Truncation required.
        match self.repr {
            UintRepr::Heap(ref mut l) => {
                l.truncate(keep);
            }
            UintRepr::Inline { ref mut len, .. } => {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "keep < INLINE_LIMBS fits safely in u8"
                )]
                let new_len = keep as u8;
                *len = new_len;
            }
        }

        if rem != 0 {
            let mask = (1_usize.wrapping_shl(rem as u32)).wrapping_sub(1);
            // SAFETY: rem != 0 implies keep >= 1, and truncation set the active
            // length to keep.
            let last = unsafe { self.limbs_mut().last_mut().unwrap_unchecked() };
            *last &= mask;
        }
        self.normalize();
        self
    }
}
