//! Multiplication by powers of two in the Fermat coefficient ring.
//!
//! Contributes the in-place shifts to [`SsaRing`]. The out-of-place variant
//! `SsaRing::shift_from` lives in the [`shift_from`](super::shift_from)
//! module.

#![allow(
    clippy::too_many_lines,
    reason = "Unrolled hybrid shift loops use raw slice chunks"
)]

use super::{Addition, LIMB_BITS, Limb, SSA_DIRECT_SHIFT_MAX_LIMBS, SsaCarry, SsaRing};

/// Widest run of limbs the shift loops process in their unrolled four-at-a-time
/// form before falling back to the plain limb-at-a-time loop.
///
/// The unrolled form keeps four independent shift chains in flight, which pays
/// while the run stays cache-resident and stops paying once it does not. Two
/// thousand forty-eight limbs is 16 KiB on a 64-bit target, so the crossover sits
/// near a typical L1 capacity — it is a hardware-sensitive value that has not yet
/// been moved into the generated tuning profile.
///
/// Shared with [`shift_from`](super::shift_from), whose two branch loops make the
/// same trade on the same runs.
pub const UNROLLED_SHIFT_MAX_LIMBS: usize = 2048;

impl SsaRing {
    /// Computes `dst = dst * 2^shift mod (2^n + 1)` in-place.
    ///
    /// Uses the identity $2^n \equiv -1 \pmod{2^n + 1}$: after left-shifting,
    /// the high part above bit `n` is subtracted from the low part.
    /// `SSA_DIRECT_SHIFT_MAX_LIMBS` selects the target-tuned crossover between the
    /// single-loop out-of-place shift and the bulk architecture-shift path.
    ///
    /// `scratch` must have length >= `SsaRing::coeff_limbs(mod_bits)` and is used as a
    /// temporary buffer.
    ///
    /// # Safety
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `scratch.len() >= SsaRing::coeff_limbs(mod_bits)`.
    pub unsafe fn shift(dst: &mut [Limb], shift: usize, mod_bits: usize, scratch: &mut [Limb]) {
        let ml = Self::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);
        let full_period = mod_bits.wrapping_mul(2);
        debug_assert!(full_period > 0, "a Fermat shift period is nonzero");
        let reduced_shift = Self::reduce_mod_period(shift, full_period);
        // SAFETY: caller guarantees dst.len() >= cl and scratch.len() >= cl.
        if reduced_shift == 0 || unsafe { dst.get_unchecked(..cl) }.iter().all(|l| *l == 0) {
            return;
        }
        if cl <= SSA_DIRECT_SHIFT_MAX_LIMBS {
            // SAFETY: the caller guarantees two disjoint cl-limb spans and dst is
            // canonical. The direct loop won through this measured width.
            unsafe {
                Self::shift_from(scratch, dst, reduced_shift, mod_bits);
            }
            // SAFETY: both spans contain cl limbs.
            unsafe { dst.get_unchecked_mut(..cl) }
                .copy_from_slice(unsafe { scratch.get_unchecked(..cl) });
            return;
        }

        let negate_result = reduced_shift >= mod_bits;
        let positive_shift = if negate_result {
            reduced_shift.wrapping_sub(mod_bits)
        } else {
            reduced_shift
        };
        if positive_shift == 0 {
            // SAFETY: dst has cl limbs and is canonical by contract.
            unsafe {
                Self::negate(dst, mod_bits);
            }
            return;
        }

        // The only canonical residue with a set guard is 2^n = -1. Convert it to
        // the ordinary power 2^positive_shift and apply its combined sign below.
        // SAFETY: ml < cl and both buffers have at least cl limbs.
        let guard = unsafe { *dst.get_unchecked(ml) };
        if guard == 1 {
            debug_assert!(
                // SAFETY: caller guarantees dst contains cl > ml limbs.
                unsafe { dst.get_unchecked(..ml) }.iter().all(|l| *l == 0),
                "the only canonical residue with a guard is 2^n"
            );
            // SAFETY: dst has cl limbs and 0 < positive_shift < mod_bits. Its
            // canonical guard proves that it represents exactly 2^n = -1; the
            // debug assertion documents the representation invariant without a
            // second release-mode scan.
            unsafe {
                write_shifted_guard_residue(dst, positive_shift, negate_result, mod_bits);
            }
            return;
        }

        // All other canonical residues have a zero guard. Preserve the n data bits
        // in scratch while dst receives the low half of the shifted product.
        debug_assert_eq!(
            guard, 0,
            "an ordinary canonical Fermat residue has a zero guard"
        );
        // SAFETY: caller guarantees scratch and dst each have cl > ml limbs.
        unsafe { scratch.get_unchecked_mut(..ml) }
            .copy_from_slice(unsafe { dst.get_unchecked(..ml) });
        let whole_limbs = positive_shift.wrapping_div(LIMB_BITS);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64, so the remainder fits in u32"
        )]
        let bit_shift = positive_shift.wrapping_rem(LIMB_BITS) as u32;
        let low_len = ml.wrapping_sub(whole_limbs);
        // Only the prefix below the whole-limb shift is zero; every remaining data
        // limb is overwritten by the copy/shift loop below.
        // SAFETY: whole_limbs < ml < cl <= dst.len().
        unsafe { dst.get_unchecked_mut(..whole_limbs) }.fill(0);
        if bit_shift == 0 {
            // SAFETY: these two ranges contain low_len limbs and do not overlap.
            unsafe { dst.get_unchecked_mut(whole_limbs..ml) }
                .copy_from_slice(unsafe { scratch.get_unchecked(..low_len) });
        } else if low_len < UNROLLED_SHIFT_MAX_LIMBS {
            let right_shift = Limb::BITS.wrapping_sub(bit_shift);
            let mut carry = 0;
            // SAFETY: low_len <= ml <= scratch.len().
            let (chunks, remainder) = unsafe { scratch.get_unchecked(..low_len) }.as_chunks::<4>();
            let mut dst_target = whole_limbs;
            for chunk in chunks {
                let [s0, s1, s2, s3] = *chunk;

                let shifted0 = s0.wrapping_shl(bit_shift) | carry;
                let shifted1 = s1.wrapping_shl(bit_shift) | s0.wrapping_shr(right_shift);
                let shifted2 = s2.wrapping_shl(bit_shift) | s1.wrapping_shr(right_shift);
                let shifted3 = s3.wrapping_shl(bit_shift) | s2.wrapping_shr(right_shift);

                // SAFETY: dst_target + 3 < whole_limbs + low_len = ml.
                unsafe {
                    *dst.get_unchecked_mut(dst_target) = shifted0;
                    *dst.get_unchecked_mut(dst_target.wrapping_add(1)) = shifted1;
                    *dst.get_unchecked_mut(dst_target.wrapping_add(2)) = shifted2;
                    *dst.get_unchecked_mut(dst_target.wrapping_add(3)) = shifted3;
                }
                carry = s3.wrapping_shr(right_shift);
                dst_target = dst_target.wrapping_add(4);
            }
            for &source in remainder {
                let shifted = source.wrapping_shl(bit_shift) | carry;
                // SAFETY: dst_target < whole_limbs + low_len = ml.
                unsafe {
                    *dst.get_unchecked_mut(dst_target) = shifted;
                }
                carry = source.wrapping_shr(right_shift);
                dst_target = dst_target.wrapping_add(1);
            }
        } else {
            let right_shift = Limb::BITS.wrapping_sub(bit_shift);
            let mut carry = 0;
            for index in 0..low_len {
                // SAFETY: index < low_len <= ml <= scratch.len().
                let source = unsafe { *scratch.get_unchecked(index) };
                let shifted = source.wrapping_shl(bit_shift) | carry;
                // SAFETY: whole_limbs + index < whole_limbs + low_len = ml.
                unsafe {
                    *dst.get_unchecked_mut(whole_limbs.wrapping_add(index)) = shifted;
                }
                carry = source.wrapping_shr(right_shift);
            }
        }
        // SAFETY: ml < cl <= dst.len().
        unsafe {
            *dst.get_unchecked_mut(ml) = 0;
        }

        // SAFETY: both buffers have cl limbs, scratch still contains the original
        // data, and dst contains the complete low half with a zero guard.
        unsafe {
            subtract_discarded_high(dst, scratch, ml, whole_limbs, bit_shift);
        }
        if negate_result {
            // SAFETY: the low-minus-high reduction above produced a canonical residue.
            unsafe {
                Self::negate(dst, mod_bits);
            }
        }
    }

    /// Computes `dst = dst * sqrt(2) mod (2^n + 1)`.
    ///
    /// `sqrt(2)` is `2^(3n/4) - 2^(n/4)`: squaring that gives
    /// `2^(3n/2) + 2^(n/2) - 2 * 2^n`, and `2^n = -1` collapses the first term to
    /// `-2^(n/2)`, leaving exactly `2`. So the ring always contains a square root
    /// of two, reachable with two shifts and one subtraction.
    ///
    /// Requires `4 | n`, which every ring the planner emits satisfies: `n` is
    /// aligned to at least `LIMB_BITS`.
    ///
    /// # Safety
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `scratch.len() >= 2 * SsaRing::coeff_limbs(mod_bits)`.
    pub unsafe fn mul_sqrt2(dst: &mut [Limb], mod_bits: usize, scratch: &mut [Limb]) {
        debug_assert!(
            mod_bits.is_multiple_of(4),
            "a Fermat ring with a square root of two has a width divisible by four"
        );
        let cl = Self::mod_limbs(mod_bits).wrapping_add(1);
        let (staged, shift_scratch) = scratch.split_at_mut(cl);
        let quarter = mod_bits.wrapping_shr(2);

        // staged = dst * 2^(n/4), then dst = dst * 2^(3n/4), then dst -= staged.
        // SAFETY: both spans hold cl limbs, they are disjoint halves of `scratch`
        // and `dst`, and the shift is already below the 2n period.
        unsafe {
            Self::shift_from(staged, dst, quarter, mod_bits);
        }
        // SAFETY: dst and shift_scratch are disjoint cl-limb spans.
        unsafe {
            Self::shift(dst, quarter.wrapping_mul(3), mod_bits, shift_scratch);
        }
        // SAFETY: staged is disjoint from dst and both carry cl limbs.
        unsafe {
            Self::sub_in_place(dst, staged, mod_bits);
        }
    }

    /// Computes `dst = dst * 2^(shift_half / 2) mod (2^n + 1)`.
    ///
    /// Twist exponents are carried in half-bit units so the negacyclic pre-twist
    /// stays available when `2n / transform_len` is odd. Without it the planner has
    /// to round the inner ring up to the next whole multiple of `transform_len`,
    /// which at the widest operands doubles the pointwise work. An odd `shift_half`
    /// is exactly one extra [`Self::mul_sqrt2`].
    ///
    /// # Safety
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `scratch.len() >= 2 * SsaRing::coeff_limbs(mod_bits)`.
    pub unsafe fn shift_half(
        dst: &mut [Limb],
        shift_half: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
    ) {
        // SAFETY: scratch covers 2 * cl limbs, so its first half satisfies the
        // whole-bit shift's contract.
        unsafe {
            Self::shift(dst, shift_half.wrapping_shr(1), mod_bits, scratch);
        }
        if shift_half.is_multiple_of(2) {
            return;
        }
        // SAFETY: the caller's 2 * cl scratch is exactly what the factor needs.
        unsafe {
            Self::mul_sqrt2(dst, mod_bits, scratch);
        }
    }
}

/// Subtracts the discarded high half of an in-place Fermat shift.
///
/// The high part is `x >> (n - shift)`. It is built at the start of `scratch`
/// and subtracted from the already-written low half because `2^n = -1` in the
/// coefficient ring. A borrow is canonicalized modulo `2^n + 1`.
///
/// # Safety
/// - `dst` and `scratch` each contain at least `ml + 1` limbs.
/// - `scratch[..ml]` is the original ordinary residue and `dst[..ml]` is its
///   shifted low half with a zero guard.
/// - `whole_limbs < ml` and `bit_shift < Limb::BITS`.
unsafe fn subtract_discarded_high(
    dst: &mut [Limb],
    scratch: &mut [Limb],
    ml: usize,
    whole_limbs: usize,
    bit_shift: u32,
) {
    let high_len = whole_limbs.wrapping_add(usize::from(bit_shift != 0));
    let high_start = ml.wrapping_sub(high_len);
    if bit_shift == 0 {
        for index in 0..high_len {
            // SAFETY: high_start + index < ml and index < high_len <= ml.
            unsafe {
                *scratch.get_unchecked_mut(index) =
                    *scratch.get_unchecked(high_start.wrapping_add(index));
            }
        }
    } else {
        let right_shift = Limb::BITS.wrapping_sub(bit_shift);
        for index in 0..high_len {
            let source_index = high_start.wrapping_add(index);
            // SAFETY: source_index < ml; the next limb is read only when it exists.
            let low = unsafe { *scratch.get_unchecked(source_index) }.wrapping_shr(right_shift);
            let high = if source_index.wrapping_add(1) < ml {
                // SAFETY: this branch proves source_index + 1 < ml <= scratch.len().
                unsafe { *scratch.get_unchecked(source_index.wrapping_add(1)) }
                    .wrapping_shl(bit_shift)
            } else {
                0
            };
            // SAFETY: index < high_len <= ml.
            unsafe {
                *scratch.get_unchecked_mut(index) = low | high;
            }
        }
    }
    // The high part occupies only high_len limbs. Subtract that active prefix,
    // then propagate its borrow through the untouched high destination limbs.
    // SAFETY: both ranges contain exactly high_len <= ml limbs.
    let borrow =
        Addition::sub_slice_in_place(unsafe { dst.get_unchecked_mut(..high_len) }, unsafe {
            scratch.get_unchecked(..high_len)
        });
    // SAFETY: high_len <= ml < dst.len().
    let final_borrow =
        borrow != 0 && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(high_len..ml) });
    if final_borrow {
        // SAFETY: the wrapped n-limb subtraction borrowed and dst has ml + 1
        // limbs, so adding the missing +1 canonicalizes it modulo 2^n + 1.
        unsafe {
            SsaCarry::correct_wrapped_shift_difference(dst, ml);
        }
    }
}

/// Writes the shifted image of the canonical guard residue `2^n = -1`.
///
/// # Safety
/// - `dst` contains at least `SsaRing::coeff_limbs(mod_bits)` limbs.
/// - `0 < positive_shift < mod_bits`.
unsafe fn write_shifted_guard_residue(
    dst: &mut [Limb],
    positive_shift: usize,
    negate_result: bool,
    mod_bits: usize,
) {
    let ml = SsaRing::mod_limbs(mod_bits);
    let cl = ml.wrapping_add(1);
    // SAFETY: the caller guarantees dst contains the full coefficient span.
    unsafe { dst.get_unchecked_mut(..cl) }.fill(0);
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "positive_shift modulo LIMB_BITS is below 64 and therefore fits in u32"
    )]
    let bit_index = positive_shift.wrapping_rem(LIMB_BITS) as u32;
    // SAFETY: 0 < positive_shift < mod_bits, so the index is below ml < cl.
    unsafe {
        *dst.get_unchecked_mut(positive_shift.wrapping_div(LIMB_BITS)) =
            1_usize.wrapping_shl(bit_index);
    }
    if !negate_result {
        // SAFETY: dst contains the canonical value 2^positive_shift.
        unsafe {
            SsaRing::negate(dst, mod_bits);
        }
    }
}
