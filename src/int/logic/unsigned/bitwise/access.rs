//! Individual-bit access, mutation, range extraction, and forward searches.

#![expect(
    unsafe_code,
    reason = "Source-bounded ranges and validated limb indices prove initialized access, disjoint range copies, and complete destination commitment"
)]

use core::ptr::copy_nonoverlapping;

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

impl InternalMpUint {
    /// Returns the bit at position `bit` (0-indexed, LSB first).
    #[must_use]
    pub fn get_bit(&self, bit: usize) -> bool {
        let limb_idx = bit.wrapping_div(LIMB_BITS);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The remainder is below LIMB_BITS <= 64, fitting u32 on 16-, 32-, and 64-bit targets"
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
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The remainder is below LIMB_BITS <= 64, fitting u32 on 16-, 32-, and 64-bit targets"
        )]
        let bit_in_limb = bit.wrapping_rem(LIMB_BITS) as u32;
        let mut result = self.clone();
        if result.limbs().len() <= limb_idx {
            // SAFETY: limb_idx = bit / LIMB_BITS <= usize::MAX / 16,
            // so limb_idx + 1 fits every supported pointer width.
            result.resize(unsafe { limb_idx.unchecked_add(1) });
        }
        // SAFETY: limb_idx < result.limbs().len() by construction after resize.
        unsafe {
            let dst = result.limbs_mut().as_mut_ptr().add(limb_idx);
            if value {
                *dst |= (1_usize).wrapping_shl(bit_in_limb);
            } else {
                *dst &= !((1_usize).wrapping_shl(bit_in_limb));
                result.normalize();
            }
        }
        // Setting a bit preserves the old nonzero top or initializes the new
        // highest slot with a nonzero bit; only clearing can need normalization.
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
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The remainder is below LIMB_BITS <= 64, fitting u32 on 16-, 32-, and 64-bit targets"
        )]
        let bit_offset = start.wrapping_rem(LIMB_BITS) as u32;
        // SAFETY: the entry guard proves start < end and start_limb < len.
        let width_bits = unsafe { end.unchecked_sub(start) };
        // SAFETY: start_limb < limbs.len(), so this initialized suffix exists.
        let source = unsafe { limbs.get_unchecked(start_limb..) };
        let requested_len = width_bits.div_ceil(LIMB_BITS);
        // Bits beyond the source are zero and need no physical storage.
        let result_len = requested_len.min(source.len());
        let mut result = Self::with_capacity(result_len);
        let mut pending = result.prepare_limb_write(result_len);
        let dst = pending.as_mut_ptr();
        if bit_offset == 0 {
            // SAFETY: result_len <= source.len(); the fresh, aligned output
            // reserves result_len limbs and cannot overlap the source.
            unsafe {
                copy_nonoverlapping(source.as_ptr(), dst, result_len);
            }
        } else {
            // SAFETY: both spans cover result_len limbs and are disjoint.
            // The source is initialized; the kernel writes every destination
            // limb. bit_offset is strictly between zero and Limb::BITS.
            unsafe {
                if result_len <= INLINE_LIMBS {
                    let _ = ArchKernels::rshift_into_small_unchecked(
                        dst,
                        source.as_ptr(),
                        result_len,
                        bit_offset,
                    );
                } else {
                    let _ = ArchKernels::rshift_into_unchecked(
                        dst,
                        source.as_ptr(),
                        result_len,
                        bit_offset,
                    );
                }
            }
            if result_len < source.len() {
                // The retained top limb also receives bits from the next
                // source limb, which the bounded shift kernel does not read.
                // SAFETY: result_len > 0 since start < end; this branch
                // proves source[result_len] exists. The shift is in 1..W.
                unsafe {
                    let top = dst.add(result_len.unchecked_sub(1));
                    *top |=
                        *source.get_unchecked(result_len) << Limb::BITS.unchecked_sub(bit_offset);
                }
            }
        }
        // SAFETY: the copy or shift initializes every result slot before
        // committing the length, and the optional merge writes only the top.
        let output = unsafe { pending.commit() };
        let remaining = width_bits.rem_euclid(LIMB_BITS);
        if remaining != 0 && result_len == requested_len {
            // SAFETY: remaining < LIMB_BITS, and result_len is nonzero.
            // The committed last limb is initialized and exclusively borrowed.
            unsafe {
                *output.last_mut().unwrap_unchecked() &=
                    Limb::MAX >> LIMB_BITS.unchecked_sub(remaining);
            }
        }
        result.normalize();
        result
    }

    /// Finds the next set bit at or after position `from`.
    ///
    /// # Panics
    /// Panics if the next set bit's position cannot be represented by `usize`.
    #[must_use]
    pub fn find_next_set_bit(&self, from: usize) -> Option<usize> {
        let start_limb = from.wrapping_div(LIMB_BITS);
        let limbs = self.limbs();
        if start_limb >= limbs.len() {
            return None;
        }
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The remainder is below LIMB_BITS <= 64, fitting u32 on 16-, 32-, and 64-bit targets"
        )]
        let start_bit = from.wrapping_rem(LIMB_BITS) as u32;

        // SAFETY: start_limb < limbs.len(), checked above.
        let first_limb = unsafe { *limbs.get_unchecked(start_limb) };
        // SAFETY: start_bit < Limb::BITS, so the shifted one is positive
        // and subtracting one gives the exact lower-start_bit mask.
        let first_mask = !unsafe { (1_usize << start_bit).unchecked_sub(1) };
        let masked = first_limb & first_mask;
        #[expect(
            clippy::as_conversions,
            reason = "trailing_zeros <= LIMB_BITS <= usize::BITS <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
        )]
        let trailing_zeros = masked.trailing_zeros() as usize;
        if trailing_zeros != LIMB_BITS {
            // SAFETY: start_limb = from / LIMB_BITS bounds its product by from;
            // trailing_zeros < LIMB_BITS stays inside this representable aligned
            // window because LIMB_BITS divides usize::MAX + 1 on every target.
            return Some(unsafe {
                start_limb
                    .unchecked_mul(LIMB_BITS)
                    .unchecked_add(trailing_zeros)
            });
        }

        // SAFETY: start_limb < limbs.len() bounds its successor by that length.
        let next_limb = unsafe { start_limb.unchecked_add(1) };
        for index in next_limb..limbs.len() {
            // SAFETY: index is in start_limb + 1 .. limbs.len().
            let limb = unsafe { *limbs.get_unchecked(index) };
            #[expect(
                clippy::as_conversions,
                reason = "trailing_zeros <= LIMB_BITS <= usize::BITS <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let next_trailing_zeros = limb.trailing_zeros() as usize;
            if next_trailing_zeros != LIMB_BITS {
                // A later addressable limb can exceed usize's bit-index range;
                // only this final position needs checking, outside the scan.
                #[expect(
                    clippy::unwrap_in_result,
                    reason = "None means no matching set bit; an existing bit with an unrepresentable position must trigger the documented boundary panic"
                )]
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("set bit position must fit in usize");
                // LIMB_BITS is 16, 32, or 64, so base has log2(LIMB_BITS)
                // zero low bits. OR adds next_trailing_zeros < LIMB_BITS
                // within the checked aligned window without overflow.
                return Some(base | next_trailing_zeros);
            }
        }
        None
    }

    /// Finds the next zero bit at or after position `from`.
    ///
    /// Because unlimited integers have infinite zero bits beyond the current
    /// storage, this always returns a value.
    ///
    /// # Panics
    /// Panics if the next zero bit's position cannot be represented by `usize`.
    #[must_use]
    pub fn find_next_zero_bit(&self, from: usize) -> usize {
        let start_limb = from.wrapping_div(LIMB_BITS);
        let limbs = self.limbs();

        if start_limb >= limbs.len() {
            // The requested bit already lies in the implicit zero extension.
            return from;
        }
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The remainder is below LIMB_BITS <= 64, fitting u32 on 16-, 32-, and 64-bit targets"
        )]
        let start_bit = from.wrapping_rem(LIMB_BITS) as u32;
        // SAFETY: start_bit < Limb::BITS makes the shifted one positive.
        let first_mask = unsafe { (1_usize << start_bit).unchecked_sub(1) };

        // SAFETY: start_limb < limbs.len(), checked above.
        let first_limb = unsafe { *limbs.get_unchecked(start_limb) };
        let masked = first_limb | first_mask;
        #[expect(
            clippy::as_conversions,
            reason = "trailing_zeros <= LIMB_BITS <= usize::BITS <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
        )]
        let trailing_zeros = (!masked).trailing_zeros() as usize;
        if trailing_zeros != LIMB_BITS {
            // SAFETY: start_limb * LIMB_BITS <= from and trailing_zeros
            // lies within this aligned window, which divides usize::MAX + 1.
            return unsafe {
                start_limb
                    .unchecked_mul(LIMB_BITS)
                    .unchecked_add(trailing_zeros)
            };
        }

        // SAFETY: start_limb < limbs.len() bounds its successor by len.
        let next_limb = unsafe { start_limb.unchecked_add(1) };
        for index in next_limb..limbs.len() {
            // SAFETY: index is in start_limb + 1 .. limbs.len().
            let limb = unsafe { *limbs.get_unchecked(index) };
            #[expect(
                clippy::as_conversions,
                reason = "trailing_zeros <= LIMB_BITS <= usize::BITS <= usize::MAX across 16-, 32-, and 64-bit targets; lossless conversion to usize."
            )]
            let next_trailing_zeros = (!limb).trailing_zeros() as usize;
            if next_trailing_zeros != LIMB_BITS {
                let base = index
                    .checked_mul(LIMB_BITS)
                    .expect("zero bit position must fit in usize");
                // The checked base has zero low bits and the offset is below
                // LIMB_BITS, so OR combines the disjoint fields exactly.
                return base | next_trailing_zeros;
            }
        }

        // All remaining stored bits are set. The first implicit zero is at
        // the complete limb width, strictly above from by the entry guard.
        limbs
            .len()
            .checked_mul(LIMB_BITS)
            .expect("zero bit position must fit in usize")
    }
}
