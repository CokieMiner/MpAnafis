//! Multiplication by powers of two in the Fermat coefficient ring.
//!
//! In-place twiddles retain only the high digits needed for Fermat reduction.
//! Square-root twists combine the arbitrary exponent with a fixed half-ring factor.

#![expect(
    unsafe_code,
    reason = "Complete disjoint coefficients and reduced exponents bound the shift windows, saved high digits, and guard corrections"
)]

use core::{mem::MaybeUninit, ptr::copy_nonoverlapping, slice::from_raw_parts};

use super::{ArchKernels, LIMB_BITS, Limb, SSA_DIRECT_SHIFT_MAX_LIMBS, SsaCarry, SsaRing};

impl SsaRing {
    /// Computes `dst = dst * 2^reduced_shift mod (2^n + 1)` in-place without staging
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
    /// - `reduced_shift < 2 * mod_bits`. Callers establish this once at their
    ///   boundary with `reduce_mod_period` and maintain it with conditional
    ///   subtraction inside twiddle recurrences.
    #[expect(
        clippy::too_many_lines,
        reason = "the fused in-place shift keeps its three phases in one coefficient sweep"
    )]
    pub unsafe fn shift_in_place(
        dst: &mut [Limb],
        reduced_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
    ) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: ml=mod_bits/LIMB_BITS, LIMB_BITS>=16, so ml+1 fits usize.
        let cl = unsafe { ml.unchecked_add(1) };
        debug_assert!(
            reduced_shift < mod_bits.saturating_mul(2),
            "reduced shift contract"
        );
        if reduced_shift == 0 {
            return;
        }
        if reduced_shift == mod_bits {
            // SAFETY: reduced_shift == mod_bits represents multiplication by 2^mod_bits = -1,
            // which is direct in-place negation without scratch staging or shifting.
            unsafe {
                Self::negate(dst, mod_bits);
            }
            return;
        }
        if cl <= SSA_DIRECT_SHIFT_MAX_LIMBS {
            let mut stack_buf = [MaybeUninit::<Limb>::uninit(); SSA_DIRECT_SHIFT_MAX_LIMBS];
            // SAFETY:
            // - cl <= SSA_DIRECT_SHIFT_MAX_LIMBS <= stack_buf.len().
            // - dst has length >= cl and does not overlap stack_buf.
            // - After copying cl limbs, the prefix 0..cl in stack_buf is fully initialized.
            // - `staged` covers exactly cl initialized limbs holding the semi-normalized input.
            // - reduced_shift < 2*mod_bits is established by the caller.
            unsafe {
                copy_nonoverlapping(dst.as_ptr(), stack_buf.as_mut_ptr().cast::<Limb>(), cl);
                let staged = from_raw_parts(stack_buf.as_ptr().cast::<Limb>(), cl);
                Self::shift_from(dst, staged, reduced_shift, mod_bits);
            }
            return;
        }

        let negate_result = reduced_shift >= mod_bits;
        if negate_result {
            // Staging lets shift_from write H-L*2^s directly, avoiding a
            // separate coefficient negation after the positive shift.
            // SAFETY:
            // - dst and scratch each have at least cl limbs and are disjoint.
            // - dst contains a semi-normalized residue with guard <= 1.
            // - After copying cl limbs, scratch prefix 0..cl is fully initialized.
            // - `staged` covers exactly cl initialized limbs holding the input.
            // - reduced_shift < 2 * mod_bits is established by the caller.
            unsafe {
                copy_nonoverlapping(dst.as_ptr(), scratch.as_mut_ptr(), cl);
                let staged = from_raw_parts(scratch.as_ptr(), cl);
                Self::shift_from(dst, staged, reduced_shift, mod_bits);
            }
            return;
        }

        let positive_shift = reduced_shift;
        // SAFETY: ml < cl and the caller guarantees dst has cl limbs.
        let guard = unsafe { *dst.get_unchecked(ml) };
        debug_assert!(guard <= 1, "a semi-normalized Fermat guard is at most one");

        let whole_limbs = positive_shift.div_euclid(LIMB_BITS);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "LIMB_BITS is at most 64, so the remainder fits in u32"
        )]
        let bit_shift = positive_shift.rem_euclid(LIMB_BITS) as u32;
        // positive_shift < mod_bits proves whole_limbs < ml, so every derived
        // range below stays inside the data span.
        // SAFETY: positive_shift<mod_bits implies whole_limbs<ml and
        // ceil(positive_shift/LIMB_BITS)<=ml; all three widths are exact.
        let (high_len, high_start, low_len) = unsafe {
            let high_len = whole_limbs.unchecked_add(usize::from(bit_shift != 0));
            (
                high_len,
                ml.unchecked_sub(high_len),
                ml.unchecked_sub(whole_limbs),
            )
        };

        // Phase 1: save `H = dst >> (n - s)` before the shifted write
        // overwrites its source limbs.
        if bit_shift == 0 {
            // SAFETY: both prefixes hold high_len == whole_limbs <= ml limbs.
            unsafe { scratch.get_unchecked_mut(..whole_limbs) }
                .copy_from_slice(unsafe { dst.get_unchecked(high_start..ml) });
        } else {
            // SAFETY: the spans are disjoint, both cover high_len limbs, and
            // 0 < Limb::BITS - bit_shift < Limb::BITS.
            let _ = unsafe {
                ArchKernels::rshift_into_unchecked(
                    scratch.as_mut_ptr(),
                    dst.as_ptr().add(high_start),
                    high_len,
                    Limb::BITS.unchecked_sub(bit_shift),
                )
            };
        }

        // Phase 2: write `L << s` over the data limbs. `L * 2^s < 2^n` proves
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
            // staging copy and shift into one memory pass.
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

        // Phase 3: subtract the saved high part: `L << s - H`.
        let borrow = if bit_shift == 0 {
            // SAFETY: dst prefix and scratch prefix each contain high_len == whole_limbs limbs.
            // neg_slice_into computes 0 - scratch[..whole_limbs] into dst[..whole_limbs] in one pass,
            // eliminating the zeroing fill followed by in-place subtraction from zero.
            unsafe {
                Self::neg_slice_into(
                    dst.get_unchecked_mut(..whole_limbs),
                    scratch.get_unchecked(..whole_limbs),
                )
            }
        } else {
            // The first whole_limbs shifted digits are zero. Negating their
            // saved high digits writes 0-H directly, without a separate fill.
            // SAFETY: both complete prefixes contain whole_limbs<ml elements;
            // a zero whole-limb offset intentionally gives two empty spans.
            let prefix_borrow = unsafe {
                Self::neg_slice_into(
                    dst.get_unchecked_mut(..whole_limbs),
                    scratch.get_unchecked(..whole_limbs),
                )
            };
            // SAFETY: high_len=whole_limbs+1<=ml. Its final saved digit is
            // below 2^bit_shift, so adding a borrow bit is <=2^bit_shift<=B/2.
            let subtrahend = unsafe {
                scratch
                    .get_unchecked(whole_limbs)
                    .unchecked_add(Limb::from(prefix_borrow))
            };
            // SAFETY: whole_limbs<ml, so this shifted seam digit is initialized.
            let seam = unsafe { dst.get_unchecked_mut(whole_limbs) };
            let (difference, escaped) = seam.overflowing_sub(subtrahend);
            *seam = difference;
            escaped
        };

        // SAFETY: high_len <= ml < dst.len().
        let escaped =
            borrow && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(high_len..ml) });
        if escaped {
            // SAFETY: the wrapped n-limb subtraction borrowed exactly once and
            // dst has cl > ml limbs.
            unsafe {
                let _ = SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
        } else {
            // SAFETY: ml<cl<=dst.len(); every path writes its output guard once.
            unsafe {
                *dst.get_unchecked_mut(ml) = 0;
            }
        }

        if guard != 0 {
            // SAFETY: dst is a complete coefficient for this ring, and positive_shift < mod_bits
            // gives a valid in-bounds correction bit.
            unsafe {
                Self::correct_guard_shift(dst, ml, cl, mod_bits, positive_shift, false);
            }
        }
    }

    /// Computes `2^reduced_shift * sqrt(2) * dst` by one arbitrary shift followed by
    /// the fixed half-ring operation `y * (2^(n/2) - 1)`.
    ///
    /// `sqrt(2)` is `2^(3n/4) - 2^(n/4)`, so `2^s * sqrt(2) = 2^(s+n/4) * (2^(n/2) - 1)`.
    /// One combined shift forms `y`, a half-ring rotation forms `2^(n/2) * y`,
    /// and their difference is the result. The two related shifts never run as
    /// independent generic operations.
    ///
    /// Requires `4 | n`, which every ring the planner emits satisfies: `n` is
    /// aligned to at least `LIMB_BITS`.
    ///
    /// # Safety
    /// - `dst.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `scratch.len() >= SsaRing::coeff_limbs(mod_bits)`.
    /// - `reduced_shift < 2 * mod_bits` and `4 * mod_bits` fits in `usize`.
    pub unsafe fn shift_sqrt2(
        dst: &mut [Limb],
        reduced_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
    ) {
        debug_assert!(
            mod_bits.is_multiple_of(4),
            "a Fermat ring with a square root of two has a width divisible by four"
        );
        let cl = Self::coeff_limbs(mod_bits).get();
        // SAFETY: scratch contains a complete coefficient by the caller contract.
        let (staged, _) = unsafe { scratch.split_at_mut_unchecked(cl) };
        let reduced = Self::sqrt2_shift(reduced_shift, mod_bits);
        // y = 2^(shift+n/4)*dst; sqrt(2)*2^shift*dst = 2^(n/2)*y-y.
        // SAFETY: both spans hold cl limbs, they are disjoint halves of `scratch`
        // and `dst`, and the shift is already below the 2n period.
        unsafe {
            Self::shift_from(staged, dst, reduced, mod_bits);
        }
        // SAFETY: staged is a semi-normalized complete coefficient disjoint
        // from dst. The fixed kernel handles both even and odd data-limb counts.
        unsafe {
            Self::half_ring_sub_from(dst, staged, mod_bits);
        }
    }

    /// Writes `2^reduced_shift * sqrt(2) * src` without staging an intermediate
    /// ordinary twist in the destination.
    ///
    /// # Safety
    /// The source, destination, and scratch are disjoint complete coefficients
    /// for a nonzero limb-aligned `mod_bits`. The source guard is at most one,
    /// `reduced_shift < 2 * mod_bits`, and `4 * mod_bits` fits in `usize`.
    pub unsafe fn shift_sqrt2_from(
        dst: &mut [Limb],
        src: &[Limb],
        reduced_shift: usize,
        mod_bits: usize,
        scratch: &mut [Limb],
    ) {
        let cl = Self::coeff_limbs(mod_bits).get();
        // SAFETY: scratch contains a complete coefficient by the caller contract.
        let (staged, _) = unsafe { scratch.split_at_mut_unchecked(cl) };
        let reduced = Self::sqrt2_shift(reduced_shift, mod_bits);
        // SAFETY: all three spans are disjoint complete coefficients. The
        // combined exponent is below 2n, and shift_from produces a guard <= 1.
        unsafe {
            Self::shift_from(staged, src, reduced, mod_bits);
            Self::half_ring_sub_from(dst, staged, mod_bits);
        }
    }
    /// Combines an already reduced whole-bit exponent with the square-root offset.
    /// The callers establish `shift < 2n` and representable `4n`, so the sum is
    /// below `2n + n/4` and needs at most one subtraction.
    fn sqrt2_shift(shift: usize, bits: usize) -> usize {
        // SAFETY: both callers supply a ring with checked 4*bits; the reduced
        // shift plus bits/4 is strictly below 2bits+bits/4<4bits.
        let (period, combined) = unsafe { (bits.unchecked_mul(2), shift.unchecked_add(bits >> 2)) };
        debug_assert!(shift < period, "the square-root exponent is reduced");
        if combined >= period {
            // SAFETY: this branch establishes combined>=period.
            unsafe { combined.unchecked_sub(period) }
        } else {
            combined
        }
    }
}
