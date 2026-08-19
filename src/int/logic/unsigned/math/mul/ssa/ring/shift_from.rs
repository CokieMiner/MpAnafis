//! Out-of-place Fermat ring shift: `dst = src * 2^shift mod (2^n + 1)`.
//!
//! Contributes `SsaRing::shift_from` to the ring namespace. Used by the in-place
//! shift path and by every transform twiddle and untwiddle stage, so it carries
//! most of the constant factor of the FFT.
//!
//! The entry point reduces the shift modulo the ring period and dispatches on the
//! two reduction cases, whose different memory traffic earns them separate loops:
//! `shift_nonnegated` for shifts in `[0, n)`, where the result is `low - high`, and
//! `shift_negated` for shifts in `[n, 2n)`, where that difference comes out
//! sign-flipped. `correct_guard_shift` applies the guard correction both share.

#![allow(
    clippy::too_many_lines,
    reason = "Unrolled hybrid shift-loops for the out-of-place path"
)]

use super::{Addition, ArchKernels, LIMB_BITS, Limb, SsaCarry, SsaRing};

impl SsaRing {
    /// Computes `dst = src * 2^shift mod (2^n + 1)` without an intermediate
    /// coefficient-sized copy.
    ///
    /// The ordinary-residue path generates the low part of `src << shift` and the
    /// discarded high part over their exact active ranges. Because `2^n = -1`, their
    /// subtraction is the reduced result; shifts in `[n, 2n)` negate that canonical
    /// difference.
    ///
    /// # Safety
    /// - `dst` and `src` each have at least `SsaRing::coeff_limbs(mod_bits)` limbs.
    /// - Their active spans do not overlap.
    /// - `src` is a semi-normalized Fermat residue: its guard limb is at most one.
    pub unsafe fn shift_from(dst: &mut [Limb], src: &[Limb], shift: usize, mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);
        // SAFETY: ml < cl and the caller guarantees src contains cl limbs.
        let guard = unsafe { *src.get_unchecked(ml) };
        debug_assert!(guard <= 1, "a semi-normalized Fermat guard is at most one");

        let reduced_shift = Self::reduce_mod_period(shift, mod_bits.wrapping_mul(2));
        if reduced_shift == 0 {
            // SAFETY: the caller guarantees both spans contain cl limbs.
            unsafe { dst.get_unchecked_mut(..cl) }
                .copy_from_slice(unsafe { src.get_unchecked(..cl) });
            return;
        }

        let negate_result = reduced_shift >= mod_bits;
        let positive_shift = if negate_result {
            reduced_shift.wrapping_sub(mod_bits)
        } else {
            reduced_shift
        };
        if positive_shift == 0 {
            // SAFETY: both spans contain cl limbs and do not overlap.
            unsafe { dst.get_unchecked_mut(..cl) }
                .copy_from_slice(unsafe { src.get_unchecked(..cl) });
            // SAFETY: dst contains the copied semi-normalized residue. Reducing
            // `low + guard*2^n` to `low-guard` makes it canonical before negation.
            unsafe {
                Self::normalize(dst, mod_bits);
            }
            // SAFETY: dst is canonical after the normalization above.
            unsafe {
                Self::negate(dst, mod_bits);
            }
            return;
        }

        let whole_limbs = positive_shift.wrapping_div(LIMB_BITS);
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64, so the remainder fits in u32"
        )]
        let bit_shift = positive_shift.wrapping_rem(LIMB_BITS) as u32;
        let right_shift = Limb::BITS.wrapping_sub(bit_shift);
        let low_len = ml.wrapping_sub(whole_limbs);
        let high_len = whole_limbs.wrapping_add(usize::from(bit_shift != 0));
        let high_start = ml.wrapping_sub(high_len);

        if negate_result {
            // SAFETY: all parameters are derived from the caller's bounds.
            unsafe {
                shift_negated(
                    dst,
                    src,
                    ml,
                    whole_limbs,
                    bit_shift,
                    right_shift,
                    low_len,
                    high_len,
                    high_start,
                );
            }
        } else {
            // SAFETY: all parameters are derived from the caller's bounds.
            unsafe {
                shift_nonnegated(
                    dst,
                    src,
                    ml,
                    whole_limbs,
                    bit_shift,
                    low_len,
                    high_len,
                    high_start,
                );
            }
        }

        if guard != 0 {
            // SAFETY: dst and mod_bits are from the caller; indices are in-bounds.
            unsafe {
                correct_guard_shift(dst, ml, cl, mod_bits, positive_shift, negate_result);
            }
        }
    }
}

/// Corrects the shifted Fermat residue for a non-zero semi-normalized guard.
///
/// A semi-normalized source stores `low - guard * 2^n`. After shifting and
/// modular reduction the guard is applied as a bit-level add or subtract at
/// `positive_shift`, followed by one round of canonicalization.
///
/// # Safety
/// Same preconditions as [`SsaRing::shift_from`], with `guard != 0`.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "positive_shift modulo LIMB_BITS is below 64 and therefore fits in u32"
)]
unsafe fn correct_guard_shift(
    dst: &mut [Limb],
    ml: usize,
    cl: usize,
    mod_bits: usize,
    positive_shift: usize,
    negate_result: bool,
) {
    let correction_bit = 1_usize.wrapping_shl(positive_shift.wrapping_rem(LIMB_BITS) as u32);
    let correction_index = positive_shift.wrapping_div(LIMB_BITS);
    if negate_result {
        // A semi-normalized source represents `low - guard`. Negating the
        // shifted low part therefore adds guard*2^positive_shift.
        // SAFETY: correction_index < ml < cl <= dst.len().
        let (corrected, carry) =
            unsafe { *dst.get_unchecked(correction_index) }.overflowing_add(correction_bit);
        // SAFETY: correction_index is in the data span.
        unsafe {
            *dst.get_unchecked_mut(correction_index) = corrected;
        }
        if carry {
            // SAFETY: the suffix remains inside the complete coefficient.
            let _ = SsaCarry::propagate_carry(unsafe {
                dst.get_unchecked_mut(correction_index.wrapping_add(1)..cl)
            });
        }
        // SAFETY: adding the one-bit guard correction produces a value
        // below 2*(2^n+1), so one canonical reduction is sufficient.
        unsafe {
            SsaRing::normalize(dst, mod_bits);
        }
    } else {
        // The positive shifted value is low*2^s - guard*2^s. Subtract the
        // guard bit through the full coefficient; if storage underflows,
        // add 2^n+1 to recover the canonical ring representative.
        // SAFETY: correction_index < ml < cl <= dst.len().
        let (corrected, borrow) =
            unsafe { *dst.get_unchecked(correction_index) }.overflowing_sub(correction_bit);
        // SAFETY: correction_index is in the data span.
        unsafe {
            *dst.get_unchecked_mut(correction_index) = corrected;
        }
        // SAFETY: the suffix remains inside the complete coefficient.
        let escaped = borrow
            && SsaCarry::propagate_borrow(unsafe {
                dst.get_unchecked_mut(correction_index.wrapping_add(1)..cl)
            });
        if escaped {
            // The cl-limb wrap is corrected by adding both terms of
            // 2^n+1: increment the complete slot, then its guard.
            // SAFETY: caller guarantees dst contains cl limbs.
            let _ = SsaCarry::propagate_carry(unsafe { dst.get_unchecked_mut(..cl) });
            // SAFETY: ml < cl <= dst.len().
            let guard_slot = unsafe { dst.get_unchecked_mut(ml) };
            *guard_slot = guard_slot.wrapping_add(1);
        }
    }
}

// ── Shifts in [0, n): low - high ─────────────────────────────────────────────

/// Applies the non-negated shift path: the low part is written at the
/// destination suffix and the high part is subtracted from the prefix.
///
/// # Safety
/// Same preconditions as [`SsaRing::shift_from`], with
/// `negate_result == false` and `positive_shift > 0`.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    reason = "standard short Fermat shift parameter naming"
)]
unsafe fn shift_nonnegated(
    dst: &mut [Limb],
    src: &[Limb],
    ml: usize,
    whole_limbs: usize,
    bit_shift: u32,
    low_len: usize,
    high_len: usize,
    high_start: usize,
) {
    // SAFETY: positive_shift < mod_bits proves whole_limbs < ml.
    unsafe { dst.get_unchecked_mut(..whole_limbs) }.fill(0);
    if bit_shift == 0 {
        // SAFETY: the source prefix and destination suffix each contain
        // low_len limbs and the caller guarantees they do not overlap.
        unsafe { dst.get_unchecked_mut(whole_limbs..ml) }
            .copy_from_slice(unsafe { src.get_unchecked(..low_len) });
    } else {
        // SAFETY: low_len <= ml, whole_limbs + low_len = ml, dst has cl > ml limbs,
        // src has cl > ml limbs, 0 < bit_shift < LIMB_BITS, and spans are non-overlapping.
        let _ = unsafe {
            ArchKernels::lshift_into_unchecked(
                dst.as_mut_ptr().add(whole_limbs),
                src.as_ptr(),
                low_len,
                bit_shift,
            )
        };
    }

    let borrow = if bit_shift == 0 {
        // SAFETY: dst contains the complete writable high_len-limb prefix.
        let low = unsafe { dst.get_unchecked_mut(..high_len) };
        // SAFETY: src[high_start..ml] contains exactly high_len initialized limbs.
        let high = unsafe { src.get_unchecked(high_start..ml) };
        Addition::sub_slice_in_place(low, high) != 0
    } else {
        let kernel = ArchKernels::selected_sub_shifted_high_limbs_unchecked();
        // SAFETY: dst and src are disjoint complete coefficients. The selected
        // destination prefix and source suffix each contain `high_len` limbs,
        // `bit_shift` is non-zero and below Limb::BITS, and the initial borrow
        // is zero. The kernel defines the limb above the source span as zero,
        // so it cannot consume the separately corrected Fermat guard.
        unsafe {
            kernel(
                dst.as_mut_ptr(),
                src.as_ptr().add(high_start),
                high_len,
                bit_shift,
                0,
            ) != 0
        }
    };
    let final_borrow =
        // SAFETY: The bounds are correct by construction and checked earlier.
        borrow && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(high_len..ml) });
    // SAFETY: ml < cl <= dst.len().
    unsafe {
        *dst.get_unchecked_mut(ml) = 0;
    }
    if final_borrow {
        // SAFETY: the wrapped n-limb subtraction borrowed and dst has cl > ml
        // limbs, so adding the missing +1 canonicalizes it modulo 2^n + 1.
        unsafe {
            SsaCarry::correct_wrapped_shift_difference(dst, ml);
        }
    }
}

// ── Shifts in [n, 2n): the sign-flipped difference ──────────────────────────────

/// Applies the negated shift path: `high * 2^positive_shift` (modulo sign flip)
/// is written at the start and the low part is subtracted.
///
/// # Safety
/// Same preconditions as [`SsaRing::shift_from`], with
/// `negate_result == true` and `positive_shift > 0`.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    reason = "standard short Fermat shift parameter naming"
)]
unsafe fn shift_negated(
    dst: &mut [Limb],
    src: &[Limb],
    ml: usize,
    whole_limbs: usize,
    bit_shift: u32,
    right_shift: u32,
    low_len: usize,
    high_len: usize,
    high_start: usize,
) {
    if bit_shift == 0 {
        // SAFETY: both ranges contain high_len limbs, end exactly at ml,
        // and the caller guarantees the source and destination disjoint.
        unsafe { dst.get_unchecked_mut(..high_len) }
            .copy_from_slice(unsafe { src.get_unchecked(high_start..ml) });
    } else {
        // SAFETY: high_len <= ml, high_start + high_len = ml, dst has cl > ml limbs,
        // src has cl > ml limbs, 0 < right_shift < LIMB_BITS, and spans are non-overlapping.
        let _ = unsafe {
            ArchKernels::rshift_into_unchecked(
                dst.as_mut_ptr(),
                src.as_ptr().add(high_start),
                high_len,
                right_shift,
            )
        };
    }
    // SAFETY: high_len <= ml < cl <= dst.len().
    unsafe { dst.get_unchecked_mut(high_len..ml) }.fill(0);
    let borrow = if bit_shift == 0 {
        // SAFETY: dst[whole_limbs..ml] contains exactly low_len writable limbs.
        let high = unsafe { dst.get_unchecked_mut(whole_limbs..ml) };
        // SAFETY: src contains the matching initialized low_len-limb prefix.
        let low = unsafe { src.get_unchecked(..low_len) };
        Addition::sub_slice_in_place(high, low) != 0
    } else {
        let mut low_carry = 0;
        let mut low_borrow = false;
        // SAFETY: low_len <= ml <= src.len().
        let (chunks, remainder) = unsafe { src.get_unchecked(..low_len) }.as_chunks::<4>();
        let mut target = whole_limbs;
        for chunk in chunks {
            let [s0, s1, s2, s3] = *chunk;

            let ls0 = s0.wrapping_shl(bit_shift) | low_carry;
            let c0 = s0.wrapping_shr(right_shift);
            let ls1 = s1.wrapping_shl(bit_shift) | c0;
            let c1 = s1.wrapping_shr(right_shift);
            let ls2 = s2.wrapping_shl(bit_shift) | c1;
            let c2 = s2.wrapping_shr(right_shift);
            let ls3 = s3.wrapping_shl(bit_shift) | c2;
            low_carry = s3.wrapping_shr(right_shift);

            // SAFETY: target + 3 < whole_limbs + low_len = ml < cl <= dst.len().
            unsafe {
                let m0 = *dst.get_unchecked(target);
                let (p0, u0a) = m0.overflowing_sub(ls0);
                let (r0, u0b) = p0.overflowing_sub(Limb::from(low_borrow));
                *dst.get_unchecked_mut(target) = r0;

                let m1 = *dst.get_unchecked(target.wrapping_add(1));
                let (p1, u1a) = m1.overflowing_sub(ls1);
                let (r1, u1b) = p1.overflowing_sub(Limb::from(u0a | u0b));
                *dst.get_unchecked_mut(target.wrapping_add(1)) = r1;

                let m2 = *dst.get_unchecked(target.wrapping_add(2));
                let (p2, u2a) = m2.overflowing_sub(ls2);
                let (r2, u2b) = p2.overflowing_sub(Limb::from(u1a | u1b));
                *dst.get_unchecked_mut(target.wrapping_add(2)) = r2;

                let m3 = *dst.get_unchecked(target.wrapping_add(3));
                let (p3, u3a) = m3.overflowing_sub(ls3);
                let (r3, u3b) = p3.overflowing_sub(Limb::from(u2a | u2b));
                *dst.get_unchecked_mut(target.wrapping_add(3)) = r3;
                low_borrow = u3a | u3b;
            }
            target = target.wrapping_add(4);
        }
        for &source in remainder {
            let low_shifted = source.wrapping_shl(bit_shift) | low_carry;
            low_carry = source.wrapping_shr(right_shift);
            // SAFETY: target < whole_limbs + low_len = ml < cl <= dst.len().
            let minuend = unsafe { *dst.get_unchecked(target) };
            let (partial, underflow_a) = minuend.overflowing_sub(low_shifted);
            let (result, underflow_b) = partial.overflowing_sub(Limb::from(low_borrow));
            // SAFETY: target < ml < cl <= dst.len().
            unsafe {
                *dst.get_unchecked_mut(target) = result;
            }
            low_borrow = underflow_a | underflow_b;
            target = target.wrapping_add(1);
        }
        low_borrow
    };
    // SAFETY: ml < cl <= dst.len().
    unsafe {
        *dst.get_unchecked_mut(ml) = 0;
    }
    if borrow {
        // SAFETY: the n-limb high-minus-low subtraction borrowed exactly
        // once, so adding 2^n+1 produces its canonical Fermat residue.
        unsafe {
            SsaCarry::correct_wrapped_shift_difference(dst, ml);
        }
    }
}
