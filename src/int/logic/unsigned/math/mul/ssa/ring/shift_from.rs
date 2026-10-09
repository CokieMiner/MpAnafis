//! Out-of-place Fermat ring shift: `dst = src * 2^shift mod (2^n + 1)`.
//!
//! Exponents below n write L*2^s-H; exponents above n write H-L*2^s.
//! The guard correction is applied once after the selected digit traversal.

#![expect(
    unsafe_code,
    reason = "Reduced exponents and complete disjoint coefficients bound the complementary shift windows and correction digits"
)]

use core::mem::MaybeUninit;

use super::{
    ArchKernels, LIMB_BITS, Limb, SSA_SHIFT_BLOCK_WIDTH, SSA_SHIFT_SCALAR_THRESHOLD, SsaCarry,
    SsaRing,
};

impl SsaRing {
    /// Computes `dst = src * 2^shift mod (2^n + 1)` by traversing active digits.
    ///
    /// # Safety
    /// - `dst` and `src` each have at least `SsaRing::coeff_limbs(mod_bits)` limbs.
    /// - Their active spans do not overlap.
    /// - `src` is a semi-normalized Fermat residue: its guard limb is at most one.
    /// - `reduced_shift < 2 * mod_bits` for a positive limb-aligned ring.
    pub unsafe fn shift_from(
        dst: &mut [Limb],
        src: &[Limb],
        reduced_shift: usize,
        mod_bits: usize,
    ) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: ml=mod_bits/LIMB_BITS with LIMB_BITS>=16, so ml+1 fits usize.
        let cl = unsafe { ml.unchecked_add(1) };
        // SAFETY: ml < cl and the caller guarantees src contains cl limbs.
        let guard = unsafe { *src.get_unchecked(ml) };
        debug_assert!(guard <= 1, "a semi-normalized Fermat guard is at most one");

        debug_assert!(
            reduced_shift < mod_bits.saturating_mul(2),
            "reduced shift contract"
        );
        if reduced_shift == 0 {
            // SAFETY: the caller guarantees both spans contain cl limbs.
            unsafe { dst.get_unchecked_mut(..cl) }
                .copy_from_slice(unsafe { src.get_unchecked(..cl) });
            return;
        }

        let negate_result = reduced_shift >= mod_bits;
        let positive_shift = if negate_result {
            // SAFETY: negate_result establishes reduced_shift>=mod_bits.
            unsafe { reduced_shift.unchecked_sub(mod_bits) }
        } else {
            reduced_shift
        };
        if positive_shift == 0 {
            // SAFETY: both spans contain cl limbs and do not overlap.
            unsafe { dst.get_unchecked_mut(..cl) }
                .copy_from_slice(unsafe { src.get_unchecked(..cl) });
            // SAFETY: dst contains the complete copied semi-normalized residue.
            unsafe {
                Self::negate(dst, mod_bits);
            }
            return;
        }

        let whole_limbs = positive_shift.div_euclid(LIMB_BITS);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64, so the remainder fits in u32"
        )]
        let bit_shift = positive_shift.rem_euclid(LIMB_BITS) as u32;
        // SAFETY: bit_shift<Limb::BITS; positive_shift<mod_bits implies
        // whole_limbs<ml and high_len=ceil(positive_shift/LIMB_BITS)<=ml.
        let (right_shift, low_len, high_len, high_start) = unsafe {
            let high_len = whole_limbs.unchecked_add(usize::from(bit_shift != 0));
            (
                Limb::BITS.unchecked_sub(bit_shift),
                ml.unchecked_sub(whole_limbs),
                high_len,
                ml.unchecked_sub(high_len),
            )
        };

        if negate_result {
            // SAFETY: all parameters are derived from the caller's bounds.
            unsafe {
                Self::shift_negated(
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
                Self::shift_nonnegated(
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
                Self::correct_guard_shift(dst, ml, cl, mod_bits, positive_shift, negate_result);
            }
        }
    }

    /// Corrects the shifted Fermat residue for a non-zero semi-normalized guard.
    ///
    /// A source `d+g*2^n` represents `d-g` modulo `2^n+1`.
    /// Its shifted correction subtracts `g*2^s`, or adds it after negation.
    ///
    /// # Safety
    /// `dst` holds a complete coefficient produced by shifting an ordinary
    /// residue by `positive_shift` in a ring of `mod_bits` bits, with `ml` and
    /// `cl` matching that ring.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "positive_shift modulo LIMB_BITS is below 64 and therefore fits u32"
    )]
    pub unsafe fn correct_guard_shift(
        dst: &mut [Limb],
        ml: usize,
        cl: usize,
        mod_bits: usize,
        positive_shift: usize,
        negate_result: bool,
    ) {
        // SAFETY: the remainder is below LIMB_BITS=usize::BITS.
        let correction_bit =
            unsafe { 1_usize.unchecked_shl(positive_shift.rem_euclid(LIMB_BITS) as u32) };
        let correction_index = positive_shift.div_euclid(LIMB_BITS);
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
                    dst.get_unchecked_mut(correction_index.unchecked_add(1)..cl)
                });
            }
            // SAFETY: adding the one-bit guard correction produces a value
            // below 2*(2^n+1), so one canonical reduction is sufficient.
            unsafe {
                let _ = Self::normalize(dst, mod_bits);
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
                    dst.get_unchecked_mut(correction_index.unchecked_add(1)..cl)
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
    /// Applies the non-negated shift path: the low part is written at the
    /// destination suffix and the high part is subtracted from the prefix.
    ///
    /// # Safety
    /// Same preconditions as [`SsaRing::shift_from`], with
    /// `negate_result == false` and `positive_shift > 0`.
    #[expect(
        clippy::too_many_arguments,
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
        let borrow = if bit_shift == 0 {
            // SAFETY: the source prefix and destination suffix each contain
            // low_len limbs and the caller guarantees they do not overlap.
            unsafe { dst.get_unchecked_mut(whole_limbs..ml) }
                .copy_from_slice(unsafe { src.get_unchecked(..low_len) });
            // SAFETY: dst prefix and src suffix each contain high_len == whole_limbs limbs.
            unsafe {
                Self::neg_slice_into(
                    dst.get_unchecked_mut(..whole_limbs),
                    src.get_unchecked(high_start..ml),
                )
            }
        } else {
            // SAFETY: positive_shift < mod_bits proves whole_limbs < ml.
            unsafe { dst.get_unchecked_mut(..whole_limbs) }.fill(0);
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
        // SAFETY: high_len<=ml and dst contains ml data limbs plus a guard.
        borrow && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(high_len..ml) });
        if final_borrow {
            // SAFETY: the wrapped n-limb subtraction borrowed and dst has cl > ml
            // limbs, so adding the missing +1 canonicalizes it modulo 2^n + 1.
            unsafe {
                let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        } else {
            // SAFETY: ml<cl<=dst.len(); this is the only output-guard write.
            unsafe {
                *dst.get_unchecked_mut(ml) = 0;
            }
        }
    }

    /// Fused shift-and-subtract chain for widths below two staged SIMD blocks.
    ///
    /// # Safety
    ///
    /// `src` must contain `low_len` readable limbs and `dst` must contain the
    /// writable range `whole_limbs..whole_limbs + low_len`. `bit_shift` and
    /// `right_shift` must be complementary nonzero limb shifts.
    unsafe fn shift_sub_scalar(
        dst: &mut [Limb],
        src: &[Limb],
        whole_limbs: usize,
        bit_shift: u32,
        right_shift: u32,
        low_len: usize,
    ) -> bool {
        let mut low_carry = 0;
        let mut low_borrow = false;
        // SAFETY: the caller guarantees this complete readable prefix.
        let (chunks, remainder) = unsafe { src.get_unchecked(..low_len) }.as_chunks::<4>();
        let mut target = whole_limbs;
        for chunk in chunks {
            let [s0, s1, s2, s3] = *chunk;

            // SAFETY: both complementary counts lie strictly between zero and Limb::BITS.
            let (ls0, ls1, ls2, ls3, next_carry) = unsafe {
                (
                    s0.unchecked_shl(bit_shift) | low_carry,
                    s1.unchecked_shl(bit_shift) | s0.unchecked_shr(right_shift),
                    s2.unchecked_shl(bit_shift) | s1.unchecked_shr(right_shift),
                    s3.unchecked_shl(bit_shift) | s2.unchecked_shr(right_shift),
                    s3.unchecked_shr(right_shift),
                )
            };
            low_carry = next_carry;

            // SAFETY: target + 3 < whole_limbs + low_len <= dst.len().
            unsafe {
                let m0 = *dst.get_unchecked(target);
                let (p0, u0a) = m0.overflowing_sub(ls0);
                let (r0, u0b) = p0.overflowing_sub(Limb::from(low_borrow));
                *dst.get_unchecked_mut(target) = r0;

                let m1 = *dst.get_unchecked(target.unchecked_add(1));
                let (p1, u1a) = m1.overflowing_sub(ls1);
                let (r1, u1b) = p1.overflowing_sub(Limb::from(u0a | u0b));
                *dst.get_unchecked_mut(target.unchecked_add(1)) = r1;

                let m2 = *dst.get_unchecked(target.unchecked_add(2));
                let (p2, u2a) = m2.overflowing_sub(ls2);
                let (r2, u2b) = p2.overflowing_sub(Limb::from(u1a | u1b));
                *dst.get_unchecked_mut(target.unchecked_add(2)) = r2;

                let m3 = *dst.get_unchecked(target.unchecked_add(3));
                let (p3, u3a) = m3.overflowing_sub(ls3);
                let (r3, u3b) = p3.overflowing_sub(Limb::from(u2a | u2b));
                *dst.get_unchecked_mut(target.unchecked_add(3)) = r3;
                low_borrow = u3a | u3b;
            }
            // SAFETY: this complete four-limb chunk ends within the derived data span.
            target = unsafe { target.unchecked_add(4) };
        }
        for &source in remainder {
            // SAFETY: the caller proves both complementary counts are below Limb::BITS.
            let (low_shifted, next_carry) = unsafe {
                (
                    source.unchecked_shl(bit_shift) | low_carry,
                    source.unchecked_shr(right_shift),
                )
            };
            low_carry = next_carry;
            // SAFETY: target < whole_limbs + low_len <= dst.len().
            let minuend = unsafe { *dst.get_unchecked(target) };
            let (partial, underflow_a) = minuend.overflowing_sub(low_shifted);
            let (result, underflow_b) = partial.overflowing_sub(Limb::from(low_borrow));
            // SAFETY: the identical target bound gives one writable limb.
            unsafe {
                *dst.get_unchecked_mut(target) = result;
            }
            low_borrow = underflow_a | underflow_b;
            // SAFETY: the remainder's final increment reaches at most ml<=dst.len().
            target = unsafe { target.unchecked_add(1) };
        }
        low_borrow
    }

    /// Writes H-L*2^s after the exponent crosses the half-period.
    ///
    /// # Safety
    /// Same preconditions as [`SsaRing::shift_from`], with
    /// `negate_result == true` and `positive_shift > 0`.
    #[expect(
        clippy::too_many_arguments,
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
            // SAFETY: dst[whole_limbs..ml] and src[..low_len] each cover low_len limbs.
            let borrow = unsafe {
                Self::neg_slice_into(
                    dst.get_unchecked_mut(whole_limbs..ml),
                    src.get_unchecked(..low_len),
                )
            };
            if borrow {
                // SAFETY: the n-limb high-minus-low subtraction borrowed exactly
                // once, so adding 2^n+1 produces its canonical Fermat residue.
                unsafe {
                    let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml);
                }
            } else {
                // SAFETY: ml<cl<=dst.len(); this is the only output-guard write.
                unsafe {
                    *dst.get_unchecked_mut(ml) = 0;
                }
            }
            return;
        }

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
        // SAFETY: high_len <= ml < cl <= dst.len().
        unsafe { dst.get_unchecked_mut(high_len..ml) }.fill(0);
        let borrow = if low_len < SSA_SHIFT_SCALAR_THRESHOLD {
            // For narrow spans, fused scalar shift-subtract avoids buffer staging overhead.
            // SAFETY: the caller's complete source and destination proofs are
            // forwarded with the exact derived active lengths.
            unsafe {
                Self::shift_sub_scalar(dst, src, whole_limbs, bit_shift, right_shift, low_len)
            }
        } else {
            // Fixed-size staging bounds temporary storage. The shift initializes
            // each active block before subtraction reads it; its suffix is unused.
            let mut tmp = [MaybeUninit::<Limb>::uninit(); SSA_SHIFT_BLOCK_WIDTH];
            let mut source_offset = 0;
            let mut low_carry = 0;
            let mut low_borrow = 0;
            while source_offset < low_len {
                // SAFETY: the loop establishes source_offset<low_len.
                let block_len = unsafe { low_len.unchecked_sub(source_offset) }.min(tmp.len());
                // SAFETY: source_offset + block_len <= low_len <= src.len(); tmp
                // contains one tuning-policy block of writable limbs, the spans
                // are disjoint, and the caller proves 0 < bit_shift < Limb::BITS.
                let next_carry = unsafe {
                    ArchKernels::lshift_into_unchecked(
                        tmp.as_mut_ptr().cast::<Limb>(),
                        src.as_ptr().add(source_offset),
                        block_len,
                        bit_shift,
                    )
                };
                // Every shifted block starts with zero low bits. The carry from
                // the preceding source block occupies only that cleared field.
                // SAFETY: block_len is nonzero inside this loop.
                unsafe {
                    *tmp.as_mut_ptr().cast::<Limb>() |= low_carry;
                }

                // SAFETY: source_offset<low_len and whole_limbs+low_len=ml.
                let target_offset = unsafe { whole_limbs.unchecked_add(source_offset) };
                // SAFETY: target_offset + block_len <= whole_limbs + low_len = ml;
                // tmp contains block_len initialized limbs and does not overlap dst.
                let block_borrow = unsafe {
                    ArchKernels::sub_limbs_unchecked(
                        dst.as_mut_ptr().add(target_offset),
                        tmp.as_ptr().cast::<Limb>(),
                        block_len,
                    )
                };
                // Subtract the borrow entering this block after the vector-staged
                // subtraction. If that subtraction already borrowed, its modular
                // result is nonzero, so the two outgoing borrows cannot both be one.
                // SAFETY: the same complete writable destination block is valid.
                let incoming_borrow = unsafe {
                    ArchKernels::propagate_borrow_unchecked(
                        dst.as_mut_ptr().add(target_offset),
                        block_len,
                        low_borrow,
                    )
                };
                low_borrow = block_borrow | incoming_borrow;
                low_carry = next_carry;
                // SAFETY: block_len<=low_len-source_offset, so the next offset<=low_len.
                source_offset = unsafe { source_offset.unchecked_add(block_len) };
            }
            low_borrow != 0
        };
        if borrow {
            // SAFETY: the n-limb high-minus-low subtraction borrowed exactly
            // once, so adding 2^n+1 produces its canonical Fermat residue.
            unsafe {
                let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        } else {
            // SAFETY: ml<cl<=dst.len(); this is the only output-guard write.
            unsafe {
                *dst.get_unchecked_mut(ml) = 0;
            }
        }
    }
}
