//! Unsigned bit scanning and bit-count queries.

#![expect(
    unsafe_code,
    reason = "Normalized storage and bounded bit windows prove initialized suffix access; validated limb counts bound population sums"
)]

use super::{InternalMpUint, LIMB_BITS, Limb};

impl InternalMpUint {
    /// Returns the number of ones in the binary representation.
    ///
    /// # Panics
    /// Panics if the number of set bits cannot be represented by `usize`.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        let limbs = self.limbs();
        let mut counts = limbs.iter().map(|&limb| {
                #[expect(
                    clippy::as_conversions,
                    reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
                )]
                let count = limb.count_ones() as usize;
                count
            });
        if limbs.len() > usize::MAX.div_euclid(LIMB_BITS) {
            // Addressable limb storage can have an unrepresentable bit count
            // on narrow targets. Sparse inputs may still have a valid count.
            return counts
                .try_fold(0_usize, usize::checked_add)
                .expect("integer population count must fit in usize");
        }
        counts.fold(0_usize, |count, ones| {
            // SAFETY: each count is at most LIMB_BITS, and the length guard
            // bounds every partial sum by len * LIMB_BITS <= usize::MAX.
            unsafe { count.unchecked_add(ones) }
        })
    }

    /// Returns the number of trailing zero bits.
    ///
    /// Returns `0` when the value is zero.
    ///
    /// # Panics
    /// Panics if the first set bit's position cannot be represented by `usize`.
    #[inline]
    #[must_use]
    pub fn trailing_zeros(&self) -> usize {
        for (index, &limb) in self.limbs().iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let tz = limb.trailing_zeros() as usize;
            // trailing_zeros returns LIMB_BITS for zero; any other value means
            // the first set bit was found and the scan can return immediately.
            if tz != LIMB_BITS {
                // The scan counts limbs, not bits; only the final position
                // needs validation, outside the loop's zero-limb path.
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("set bit position must fit in usize");
                // The checked base has zero low bits and tz < LIMB_BITS,
                // so OR combines the disjoint fields without overflow.
                return base | tz;
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
        // A nonempty normalized value has a nonzero top limb. A window
        // covering every stored limb therefore needs no scan.
        if full_limbs >= limbs.len() {
            return !limbs.is_empty();
        }
        // SAFETY: the earlier return leaves full_limbs < limbs.len(), so this
        // complete initialized prefix exists without take's length truncation.
        let lower = unsafe { limbs.get_unchecked(..full_limbs) };
        if lower.iter().any(|&limb| limb != 0) {
            return true;
        }
        if rem_bits != 0 {
            // SAFETY: full_limbs < limbs.len() was established above.
            let limb = unsafe { *limbs.get_unchecked(full_limbs) };
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "The positive difference is below LIMB_BITS <= 64, fitting u32 on every supported pointer width"
            )]
            // SAFETY: this arm has 0 < rem_bits < LIMB_BITS <= 64, so the
            // difference is a valid shift and converts losslessly to u32.
            let shift = unsafe { LIMB_BITS.unchecked_sub(rem_bits) } as u32;
            let mask = Limb::MAX >> shift;
            return (limb & mask) != 0;
        }
        false
    }

    /// Finds the position of the first (least-significant) zero bit.
    ///
    /// Because an arbitrary-precision integer has conceptually infinite
    /// leading zeros, a zero bit always exists — even for zero, where
    /// bit `0` is the first zero bit.
    ///
    /// # Panics
    /// Panics if the first zero bit's position cannot be represented by `usize`.
    #[must_use]
    pub fn find_first_zero_bit(&self) -> usize {
        let limbs = self.limbs();
        for (index, &limb) in limbs.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let trailing_ones = limb.trailing_ones() as usize;
            // trailing_ones returns LIMB_BITS for !0 (all ones); any other
            // value means the first zero bit was found.
            if trailing_ones != LIMB_BITS {
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("zero bit position must fit in usize");
                // The checked base has zero low bits and the offset is below
                // LIMB_BITS, so OR combines the disjoint fields exactly.
                return base | trailing_ones;
            }
        }
        // No limbs or all limbs are all-ones; for empty (zero) the first
        // zero bit is at position 0; for an all-ones value the first zero
        // bit is at the complete stored width (conceptually beyond all limbs,
        // which is correct for unlimited precision).
        limbs
            .len()
            .checked_mul(LIMB_BITS)
            .expect("zero bit position must fit in usize")
    }

    /// Returns the index of the least significant set bit, or `None` if zero.
    ///
    /// # Panics
    /// Panics if the first set bit's position cannot be represented by `usize`.
    #[must_use]
    pub fn find_first_set_bit(&self) -> Option<usize> {
        for (index, &limb) in self.limbs().iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let trailing_zeros = limb.trailing_zeros() as usize;
            // trailing_zeros returns LIMB_BITS for zero; any other value
            // means the first set bit was found.
            if trailing_zeros != LIMB_BITS {
                #[expect(
                    clippy::unwrap_in_result,
                    reason = "None means zero; a nonzero magnitude with an unrepresentable first set-bit position must trigger the documented boundary panic"
                )]
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("set bit position must fit in usize");
                // LIMB_BITS is 16, 32, or 64, so base has log2(LIMB_BITS)
                // zero low bits. OR adds trailing_zeros < LIMB_BITS within
                // the checked aligned window without overflow.
                return Some(base | trailing_zeros);
            }
        }
        None
    }

    /// Returns the number of significant bits in the binary representation.
    /// Returns 0 if the value is zero.
    ///
    /// # Panics
    /// Panics if the significant bit count cannot be represented by `usize`.
    #[must_use]
    pub fn significant_bits(&self) -> usize {
        let limbs = self.limbs();
        if limbs.is_empty() {
            return 0;
        }
        // SAFETY: the early return proves the active slice is nonempty.
        let last_idx = unsafe { limbs.len().unchecked_sub(1) };
        // SAFETY: last_idx = limbs.len() - 1 is valid because is_empty() returned false.
        let last_limb = unsafe { *limbs.get_unchecked(last_idx) };
        // A non-empty `InternalMpUint` is normalized, so `last_limb != 0`.
        #[expect(
            clippy::as_conversions,
            reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
        )]
        let lz = last_limb.leading_zeros() as usize;
        // SAFETY: normalization makes the highest limb nonzero, hence lz < W.
        let bits_in_last = unsafe { LIMB_BITS.unchecked_sub(lz) };
        // `last_idx` is arbitrary user integer size; `last_idx*LIMB_BITS`
        // can overflow `usize` near the top (e.g. 16-bit `last_idx>4095`)
        // while the limb allocation is still addressable, so this checked
        // calculation is the validate-once allocation-size/bit-length
        // boundary and cannot be erased.
        last_idx
            .checked_mul(LIMB_BITS)
            .and_then(|lower_bits| lower_bits.checked_add(bits_in_last))
            .expect("integer bit length must fit in usize")
    }

    /// Tests whether the magnitude is below `2^bits` without computing its
    /// full bit count, which can exceed `usize` on narrow pointer targets.
    #[inline]
    #[must_use]
    pub fn fits_in_bits(&self, bits: usize) -> bool {
        let limbs = self.limbs();
        let full_limbs = bits.div_euclid(LIMB_BITS);
        if limbs.len() <= full_limbs {
            return true;
        }
        // SAFETY: len > full_limbs proves nonemptiness and bounds the
        // initialized, aligned immutable read at full_limbs. Normalization
        // makes every longer magnitude exceed the requested width. The
        // remainder is below LIMB_BITS on every pointer width.
        unsafe {
            limbs.len().unchecked_sub(1) == full_limbs
                && (*limbs.get_unchecked(full_limbs) >> bits.rem_euclid(LIMB_BITS)) == 0
        }
    }

    /// Returns the required bit width for bounded storage of this magnitude.
    #[must_use]
    pub fn required_unsigned_bits_for_bounded_storage(&self) -> usize {
        self.significant_bits().max(1)
    }

    /// Returns the number of leading zeros relative to the given width.
    #[must_use]
    pub fn leading_zeros_for_width(&self, width: usize) -> usize {
        width.saturating_sub(self.significant_bits())
    }

    /// Returns the number of leading ones within the given width.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        reason = "limb shifts and leading-one counts are bounded by LIMB_BITS <= 64 on every pointer width"
    )]
    pub fn leading_ones_for_width(&self, width: usize) -> usize {
        if width == 0 {
            return 0;
        }
        let limbs = self.limbs();
        // SAFETY: width > 0 proves the inclusive highest bit fits usize.
        let highest_bit = unsafe { width.unchecked_sub(1) };
        let top_limb_idx = highest_bit.div_euclid(LIMB_BITS);
        // SAFETY: the remainder is at most LIMB_BITS - 1 <= 63.
        let top_bits = unsafe { highest_bit.rem_euclid(LIMB_BITS).unchecked_add(1) };

        let Some(&top) = limbs.get(top_limb_idx) else {
            return 0;
        };
        // Left alignment discards bits above the requested width and fills
        // the low padding with zeros, bounding the count by top_bits.
        // SAFETY: 1 <= top_bits <= LIMB_BITS gives a shift in 0..LIMB_BITS.
        let aligned = top << unsafe { LIMB_BITS.unchecked_sub(top_bits) };
        let mut count = aligned.leading_ones() as usize;
        if count < top_bits {
            return count;
        }
        // SAFETY: the successful top-limb lookup proves this lower prefix exists.
        let lower = unsafe { limbs.get_unchecked(..top_limb_idx) };
        for &limb in lower.iter().rev() {
            let ones = limb.leading_ones() as usize;
            // SAFETY: the disjoint scanned windows lie inside width; their
            // complete count is at most width <= usize::MAX on every target.
            count = unsafe { count.unchecked_add(ones) };
            if ones != LIMB_BITS {
                break;
            }
        }

        count
    }

    /// Counts the number of trailing (least-significant) one bits.
    ///
    /// # Panics
    /// Panics if the trailing-one count cannot be represented by `usize`.
    #[must_use]
    pub fn trailing_ones(&self) -> usize {
        let limbs = self.limbs();
        for (index, &limb) in limbs.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let ones = limb.trailing_ones() as usize;
            // trailing_ones returns LIMB_BITS for !0 (all ones); any other
            // value means the first zero bit was found.
            if ones != LIMB_BITS {
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("trailing-one count must fit in usize");
                // The checked base has zero low bits and ones < LIMB_BITS,
                // so OR combines the disjoint fields without overflow.
                return base | ones;
            }
        }
        limbs
            .len()
            .checked_mul(LIMB_BITS)
            .expect("trailing-one count must fit in usize")
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
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            {
                // SAFETY: at most full_limbs complete windows are counted;
                // their aggregate is at most full_limbs * LIMB_BITS <= width.
                ones = unsafe { ones.unchecked_add(limb.count_ones() as usize) };
            }
        }

        if rem_bits != 0
            && let Some(&limb) = limbs.get(full_limbs)
        {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "rem_bits < LIMB_BITS <= 64 fits in u32"
            )]
            let rem_bits_u32 = rem_bits as u32;
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS is 16, 32, or 64 and always fits in u32"
            )]
            let limb_bits_u32 = LIMB_BITS as u32;
            // SAFETY: 0 < rem_bits_u32 < limb_bits_u32 bounds the shift.
            let shift = unsafe { limb_bits_u32.unchecked_sub(rem_bits_u32) };
            let mask = Limb::MAX >> shift;
            #[expect(
                clippy::as_conversions,
                reason = "maximum count is LIMB_BITS <= 64 <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            {
                // SAFETY: the mask limits this final count to rem_bits; the
                // complete disjoint windows therefore contain at most width ones.
                ones = unsafe { ones.unchecked_add((limb & mask).count_ones() as usize) };
            }
        }

        // SAFETY: only bits in the requested width were counted, so ones <= width.
        unsafe { width.unchecked_sub(ones) }
    }
}
