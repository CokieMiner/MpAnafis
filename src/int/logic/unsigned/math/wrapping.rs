//! Reduction and additive inverses modulo a power of two.

#![expect(
    unsafe_code,
    reason = "Retained prefixes and reserved write guards bound initialized limb access, length commits, and nonmodular index arithmetic."
)]

use core::ptr::write_bytes;

use super::{InternalMpUint, LIMB_BITS, Limb};

impl InternalMpUint {
    /// Reduces this value modulo `2^bits`, retaining its allocation.
    ///
    /// A zero width yields zero. Widths beyond the current magnitude leave it
    /// unchanged; reduction never allocates or exposes spare capacity.
    #[inline]
    #[must_use]
    pub fn apply_wrapping(mut self, bits: usize) -> Self {
        let _ = self.apply_wrapping_in_place(bits);
        self
    }

    /// Reduces modulo `2^bits` and reports whether any nonzero bit was discarded.
    ///
    /// Normalization makes every discarded complete high limb evidence of
    /// overflow. A partial last limb needs only its pre-mask value.
    #[inline]
    #[must_use]
    pub fn apply_wrapping_with_overflow(mut self, bits: usize) -> (Self, bool) {
        let overflow = self.apply_wrapping_in_place(bits);
        (self, overflow)
    }

    /// Computes `-self mod 2^bits` in place, retaining allocated capacity.
    ///
    /// Low zero limbs remain zero. The first nonzero limb is negated, higher
    /// existing limbs are complemented, and newly exposed high limbs are ones.
    /// A zero residue requires no growth, even when `bits == usize::MAX`.
    ///
    /// # Panics
    ///
    /// Panics if a nonzero result requires an unrepresentable allocation size.
    #[inline]
    #[must_use]
    pub fn apply_negate_wrapping(mut self, bits: usize) -> Self {
        let keep = bits.div_ceil(LIMB_BITS);
        let original = self.limbs();
        let old_len = original.len();
        let shared = old_len.min(keep);
        // A bounded index supplies the raw kernel directly, without deriving
        // an iterator position or constructing a byte-counted suffix view.
        let mut first = 0;
        while first < shared {
            // SAFETY: first < shared <= old_len bounds this initialized read.
            if unsafe { *original.get_unchecked(first) } != 0 {
                break;
            }
            // SAFETY: first < shared proves first + 1 <= shared <= usize::MAX.
            first = unsafe { first.unchecked_add(1) };
        }
        if first == shared {
            // The retained residue is zero, including zero-width truncation.
            // No extension or allocation is needed even for a very large width.
            self.clear();
            return self;
        }
        // SAFETY: first < shared <= keep, so first+1 is representable.
        let after_first = unsafe { first.unchecked_add(1) };
        let rem = bits.rem_euclid(LIMB_BITS);
        let mask = if rem == 0 {
            Limb::MAX
        } else {
            // SAFETY: 0 < rem < LIMB_BITS proves a valid, nonzero shift.
            Limb::MAX >> unsafe { LIMB_BITS.unchecked_sub(rem) }
        };
        let mut final_len = keep;
        if keep <= old_len {
            // No new limbs are needed. For radix B = 2^LIMB_BITS, the first
            // nonzero column becomes B-a[first], and every later column is
            // B-1-a[i] = !a[i]. Determine the normalized output length before
            // overwriting, so discarded zero columns never need to be written.
            // SAFETY: first < shared <= keep proves keep > 0.
            let top_index = unsafe { keep.unchecked_sub(1) };
            // SAFETY: keep <= old_len gives shared == keep, hence top_index
            // indexes the initialized immutable retained prefix.
            let top = unsafe { *original.get_unchecked(top_index) };
            let top_residue = if top_index == first {
                top.wrapping_neg()
            } else {
                !top
            };
            if top_residue & mask == 0 {
                final_len = top_index;
                if final_len == first {
                    // The first nonzero source limb is the partial top limb,
                    // and all its set bits lie above the requested width.
                    self.clear();
                    return self;
                }
                while final_len > after_first {
                    // SAFETY: after_first <= final_len <= top_index < keep;
                    // final_len > after_first >= 1 prevents underflow.
                    let index = unsafe { final_len.unchecked_sub(1) };
                    // SAFETY: index < keep <= old_len bounds this read.
                    if unsafe { *original.get_unchecked(index) } != Limb::MAX {
                        break;
                    }
                    final_len = index;
                }
            }
        }
        let initialized = old_len.min(final_len);
        let mut pending = self.prepare_limb_write(final_len);
        let ptr = pending.as_mut_ptr();
        // SAFETY: trimming stops at after_first = first + 1; the sole case
        // reaching first returned zero above. Growth has first < old_len.
        // Thus first < final_len and this preserved source slot is initialized.
        // Only this exclusive guard can access the destination allocation.
        // Since a[first] > 0, its native negative requires modular underflow.
        unsafe {
            *ptr.add(first) = (*ptr.add(first)).wrapping_neg();
        }
        let mut index = after_first;
        while index < initialized {
            // SAFETY: index < initialized <= old_len bounds this preserved,
            // initialized slot in the guard's aligned, exclusive allocation.
            unsafe {
                *ptr.add(index) = !*ptr.add(index);
            }
            // SAFETY: index < initialized proves index + 1 <= initialized,
            // so advancing cannot overflow on any supported pointer width.
            index = unsafe { index.unchecked_add(1) };
        }
        if final_len > old_len {
            // SAFETY: the guard reserves final_len limbs; old_len..final_len lies in
            // its writable spare capacity. Each byte set to 0xff produces a
            // valid all-ones limb on 16-, 32-, and 64-bit targets.
            unsafe {
                write_bytes(ptr.add(old_len), 0xff, final_len.unchecked_sub(old_len));
            }
        }
        // SAFETY: the zero prefix, negated limb, complemented shared suffix,
        // and all-ones extension initialize every limb in 0..final_len.
        let limbs = unsafe { pending.commit() };
        if rem != 0 && final_len == keep {
            // SAFETY: first < final_len proves final_len > 0, so subtraction
            // cannot underflow and the last initialized limb exists.
            let top_index = unsafe { final_len.unchecked_sub(1) };
            // SAFETY: top_index < final_len == limbs.len() bounds this
            // exclusive access to the committed, initialized prefix.
            *unsafe { limbs.get_unchecked_mut(top_index) } &= mask;
        }
        // Nongrowing outputs were trimmed against their exact high residues.
        // Growing outputs end in all ones, whose partial mask remains nonzero.
        // Both cases commit a normalized value without a second length update.
        self
    }

    /// Truncates initialized storage once and records discarded nonzero bits.
    ///
    /// Both owned return shapes use this kernel without moving storage through
    /// an intermediate value-and-flag tuple.
    #[inline]
    fn apply_wrapping_in_place(&mut self, bits: usize) -> bool {
        let full = bits.div_euclid(LIMB_BITS);
        let limbs = self.limbs_mut();
        let current_len = limbs.len();
        // full complete retained limbs cover the value exactly when full >=
        // current_len. Comparing limb counts avoids forming current_len *
        // LIMB_BITS, which need not fit usize on every supported pointer width.
        if full >= current_len {
            return false;
        }
        let rem = bits.rem_euclid(LIMB_BITS);
        let mut final_len = full;
        let overflow;
        let needs_trimming = if rem != 0 {
            // SAFETY: 0 < rem < LIMB_BITS proves a valid, nonzero shift.
            let mask = Limb::MAX >> unsafe { LIMB_BITS.unchecked_sub(rem) };
            // SAFETY: full < current_len bounds the partial top limb in the
            // initialized, exclusively borrowed source prefix.
            let last = unsafe { limbs.get_unchecked_mut(full) };
            let original = *last;
            *last &= mask;
            // SAFETY: full < current_len proves full + 1 <= current_len.
            let retained_len = unsafe { full.unchecked_add(1) };
            // A discarded most-significant limb is nonzero by normalization;
            // otherwise only the masked bits of this final limb can overflow.
            overflow = retained_len < current_len || original != *last;
            let is_zero = *last == 0;
            if !is_zero {
                final_len = retained_len;
            }
            // A zero partial limb is outside final_len and is never reloaded
            // by trimming; a nonzero partial limb already proves normalization.
            is_zero
        } else {
            // full < current_len discards the nonzero normalized top limb.
            overflow = true;
            true
        };
        if needs_trimming {
            while final_len != 0 {
                // SAFETY: 0 < final_len <= full < limbs.len(); subtraction
                // cannot underflow and the resulting index is initialized.
                let top_index = unsafe { final_len.unchecked_sub(1) };
                // SAFETY: top_index < final_len <= limbs.len().
                if unsafe { *limbs.get_unchecked(top_index) } != 0 {
                    break;
                }
                final_len = top_index;
            }
        }
        if final_len != current_len {
            // SAFETY: final_len <= current_len only shortens initialized
            // storage. Its last limb is nonzero, or the result is canonical zero.
            // Commit once after masking and trimming, retaining the allocation.
            unsafe {
                self.set_len(final_len);
            }
        }
        overflow
    }
}
