//! Multiplication by powers of two in the Fermat coefficient ring.
//!
//! Contributes the in-place shifts to [`SsaRing`]. The out-of-place variant
//! `SsaRing::shift_from` lives in the [`shift_from`](super::shift_from)
//! module.
//!
//! [`SsaRing::shift_in_place`] is the transform butterflies' twiddle: it
//! multiplies a semi-normalized coefficient by a power of two without staging
//! the whole coefficient, removing the scratch round trip an out-of-place
//! shift plus copy-back would pay on every butterfly. [`SsaRing::shift`] is
//! the zero-test wrapper the twist sweeps keep for sparse operands.

use super::{
    Addition, ArchKernels, LIMB_BITS, Limb, SSA_DIRECT_SHIFT_MAX_LIMBS, SsaCarry, SsaRing,
};

impl SsaRing {
    /// Computes `dst = dst * 2^shift mod (2^n + 1)` in-place.
    ///
    /// A zero coefficient is left untouched, which is what the twist sweeps on
    /// freshly split operands want: their padding tail is all zero and a shift
    /// would only rewrite it. `SSA_DIRECT_SHIFT_MAX_LIMBS` selects the
    /// target-tuned crossover between the single-loop out-of-place shift and
    /// the fused in-place path, both inside [`Self::shift_in_place`].
    ///
    /// `scratch` must have length >= `SsaRing::coeff_limbs(mod_bits)` and is used as a
    /// temporary buffer.
    ///
    /// # Safety
    /// - `mod_bits` is nonzero and `2 * mod_bits` fits in `usize`.
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `scratch.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - The active `dst` prefix contains a canonical Fermat-ring residue.
    /// - The active `dst` and `scratch` prefixes are disjoint. Exact or partial
    ///   overlap is forbidden because the fused path stages the discarded high
    ///   part in `scratch` while `dst` is overwritten.
    pub unsafe fn shift(dst: &mut [Limb], shift: usize, mod_bits: usize, scratch: &mut [Limb]) {
        let cl = Self::mod_limbs(mod_bits).wrapping_add(1);
        let full_period = mod_bits.wrapping_mul(2);
        debug_assert!(full_period > 0, "a Fermat shift period is nonzero");
        let reduced_shift = Self::reduce_mod_period(shift, full_period);
        // SAFETY: caller guarantees dst.len() >= cl.
        if reduced_shift == 0 || unsafe { dst.get_unchecked(..cl) }.iter().all(|l| *l == 0) {
            return;
        }
        // SAFETY: the zero test proved dst nonzero; the caller guarantees a
        // disjoint cl-limb scratch.
        unsafe {
            Self::shift_in_place(dst, reduced_shift, mod_bits, scratch);
        }
    }

    /// Computes `dst = dst * 2^shift mod (2^n + 1)` in-place without staging
    /// the whole coefficient.
    ///
    /// The data limbs are swept twice instead of the out-of-place shift plus
    /// copy-back three sweeps: the discarded high part `H = dst >> (n - s)` is
    /// saved into `scratch` first, `L << s` is written over the coefficient
    /// from the top limb down so every read stays below the writes, and `H` is
    /// subtracted through the low limbs using `2^n = -1`. A semi-normalized
    /// input guard is corrected after the main sweep, and a shift of half a
    /// period or more negates the result first; the negation runs *before* the
    /// guard correction so the correction's sign matches the negated path of
    /// [`Self::shift_from`].
    ///
    /// # Safety
    /// - `mod_bits` is nonzero and `2 * mod_bits` fits in `usize`.
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)` and holds a
    ///   semi-normalized residue: its guard limb is at most one.
    /// - `scratch.len() >= SsaRing::coeff_limbs(mod_bits)` and the two buffers
    ///   are disjoint.
    #[allow(
        clippy::too_many_lines,
        reason = "the fused in-place shift keeps its three phases in one coefficient sweep"
    )]
    pub unsafe fn shift_in_place(
        dst: &mut [Limb],
        shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
    ) {
        let ml = Self::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);
        let full_period = mod_bits.wrapping_mul(2);
        let reduced_shift = Self::reduce_mod_period(shift, full_period);
        if reduced_shift == 0 {
            return;
        }
        if cl <= SSA_DIRECT_SHIFT_MAX_LIMBS {
            // SAFETY: the caller guarantees two disjoint cl-limb spans and dst
            // is semi-normalized, which the out-of-place shift accepts. The
            // direct loop won through this measured width.
            unsafe {
                Self::shift_from(scratch, dst, reduced_shift, mod_bits);
            }
            // SAFETY: both spans contain cl limbs and the shift fully wrote
            // its destination.
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
        // SAFETY: ml < cl and the caller guarantees dst has cl limbs.
        let guard = unsafe { *dst.get_unchecked(ml) };
        debug_assert!(guard <= 1, "a semi-normalized Fermat guard is at most one");

        if positive_shift == 0 {
            // The reduced shift is exactly half a period: a pure negation.
            if guard != 0 {
                // SAFETY: dst has cl limbs and mod_bits matches.
                unsafe {
                    Self::normalize(dst, mod_bits);
                }
            }
            // SAFETY: dst is canonical after the optional normalization.
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
        // positive_shift < mod_bits proves whole_limbs < ml, so every derived
        // range below stays inside the data span.
        let high_len = whole_limbs.wrapping_add(usize::from(bit_shift != 0));
        let high_start = ml.wrapping_sub(high_len);
        let low_len = ml.wrapping_sub(whole_limbs);

        // Phase 1 — save `H = dst >> (n - s)` before the shifted write
        // overwrites its source limbs.
        if bit_shift == 0 {
            // SAFETY: both prefixes hold high_len <= ml limbs.
            unsafe { scratch.get_unchecked_mut(..high_len) }
                .copy_from_slice(unsafe { dst.get_unchecked(high_start..ml) });
        } else {
            // SAFETY: the spans are disjoint, both cover high_len limbs, and
            // 0 < Limb::BITS - bit_shift < Limb::BITS.
            let _ = unsafe {
                ArchKernels::rshift_into_unchecked(
                    scratch.as_mut_ptr(),
                    dst.as_ptr().add(high_start),
                    high_len,
                    Limb::BITS.wrapping_sub(bit_shift),
                )
            };
        }

        // Phase 2 — write `L << s` over the data limbs. `L * 2^s < 2^n` proves
        // the result fits the written window exactly, so the discarded carry is
        // zero and the architecture kernel applies.
        if bit_shift == 0 {
            // An overlap-safe memmove; whole_limbs + low_len = ml.
            dst.copy_within(..low_len, whole_limbs);
        } else if whole_limbs >= low_len {
            // The shifted destination suffix is disjoint from its source
            // prefix, so the vectorized kernel writes it in place directly.
            // SAFETY: dst covers ml data limbs; the source window
            // `[0, low_len)` and destination `[whole_limbs, ml)` do not
            // overlap because whole_limbs >= low_len; 0 < bit_shift < Limb::BITS.
            let _ = unsafe {
                ArchKernels::lshift_into_unchecked(
                    dst.as_mut_ptr().add(whole_limbs),
                    dst.as_ptr(),
                    low_len,
                    bit_shift,
                )
            };
        } else {
            // Shifts below half the ring width overlap their source. The
            // selected backend traverses high to low, consuming every source
            // before a higher destination store can overwrite it. This fuses
            // the former staging copy and shift into one memory pass.
            // SAFETY: dst covers ml limbs, whole_limbs + low_len = ml, and
            // 0 < bit_shift < Limb::BITS.
            let _ = unsafe {
                ArchKernels::lshift_overlapping_unchecked(
                    dst.as_mut_ptr(),
                    low_len,
                    whole_limbs,
                    bit_shift,
                )
            };
        }
        // SAFETY: whole_limbs < ml, proved from positive_shift < mod_bits.
        unsafe { dst.get_unchecked_mut(..whole_limbs) }.fill(0);

        // Phase 3 — subtract the saved high part: `L << s - H`.
        // SAFETY: both prefixes hold high_len limbs.
        let borrow =
            Addition::sub_slice_in_place(unsafe { dst.get_unchecked_mut(..high_len) }, unsafe {
                scratch.get_unchecked(..high_len)
            }) != 0;
        // SAFETY: high_len <= ml < dst.len().
        let escaped =
            borrow && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(high_len..ml) });
        // SAFETY: ml < cl <= dst.len().
        unsafe {
            *dst.get_unchecked_mut(ml) = 0;
        }
        if escaped {
            // SAFETY: the wrapped n-limb subtraction borrowed exactly once and
            // dst has cl > ml limbs.
            unsafe {
                SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        }

        if negate_result {
            // SAFETY: the phases above produced a canonical residue, which is
            // what negation requires.
            unsafe {
                Self::negate(dst, mod_bits);
            }
        }
        if guard != 0 {
            // SAFETY: dst is a complete coefficient for this ring, and the
            // negation above ran first so this correction lands with the sign
            // the value algebra requires.
            unsafe {
                Self::correct_guard_shift(dst, ml, cl, mod_bits, positive_shift, negate_result);
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
}
