//! Fermat residue classification, normalization, and fused limb arithmetic.

#![expect(
    unsafe_code,
    reason = "Admitted positive aligned rings and complete initialized coefficients bound data and guard access and exact carry corrections"
)]

use core::num::NonZeroUsize;

use super::{LIMB_BITS, Limb, SsaCarry};

/// A canonical Fermat residue's relation to the two multiplicative special
/// cases that every product path short-circuits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Residue {
    /// The additive identity: every data limb and the guard limb are zero.
    Zero,
    /// The residue `-1 == 2^n`: every data limb is zero and the guard is one.
    NegOne,
    /// Anything else, including a non-canonical guard above one.
    Ordinary,
}

/// Namespace for arithmetic in the Fermat ring `Z/(2^mod_bits + 1)`.
///
/// The whole `ring` folder contributes to this one namespace: [`arith`](self)
/// supplies the slot widths and the add/subtract/negate/normalize family,
/// [`shift`](super::shift) the in-place multiplications by a power of two, and
/// [`shift_from`](super::shift_from) their out-of-place form. Arithmetic methods
/// receive explicit ring widths because one transform nests several rings.
/// Classification uses their admitted data width; plain slice leaves use their extents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaRing;

impl SsaRing {
    /// Classifies a canonical residue with a single pass over its data limbs.
    ///
    /// Zero and `-1` have zero data and differ only in their guard.
    ///
    /// # Safety
    /// `ml` is the positive data width of the admitted ring; `coeff.len() > ml`.
    #[expect(
        clippy::inline_always,
        reason = "single-pass classification on the hot pointwise path"
    )]
    #[inline(always)]
    pub unsafe fn classify_residue(coeff: &[Limb], ml: usize) -> Residue {
        // Either special residue requires every data limb to be zero.
        // SAFETY: caller guarantees coeff.len() > ml >= 1.
        if unsafe { *coeff.get_unchecked(0) } != 0 {
            return Residue::Ordinary;
        }
        // SAFETY: ml < coeff.len() as guaranteed by the caller.
        let guard = unsafe { *coeff.get_unchecked(ml) };
        if guard > 1 {
            return Residue::Ordinary;
        }
        // SAFETY: caller guarantees coeff.len() > ml, so 1..ml is in range.
        let data_zero = unsafe { coeff.get_unchecked(1..ml) }
            .iter()
            .all(|l| *l == 0);
        if !data_zero {
            return Residue::Ordinary;
        }
        // The earlier guard bound leaves exactly the two all-zero-data cases.
        if guard == 0 {
            Residue::Zero
        } else {
            Residue::NegOne
        }
    }

    /// Replaces `sum` and `difference` with `(sum + source, sum - source)` modulo
    /// `2^n + 1` in one fused data-limb pass.
    ///
    /// Inputs and outputs are semi-normalized: the guard limb is at most one, but
    /// a guard of one may accompany nonzero data limbs. Writing a slot as
    /// `low + guard * 2^n`, its ring value is `low - guard`; the post-kernel
    /// corrections below reduce the signed guard coefficient to `{0, 1}` without
    /// scanning or fully canonicalizing the data limbs.
    ///
    /// # Safety
    /// - Both slices and `source` cover at least `coeff_limbs(mod_bits)` limbs.
    /// - `source` is disjoint from `sum` and is either disjoint from `difference`
    ///   or points to the exact same span.
    pub unsafe fn add_sub(
        sum: &mut [Limb],
        difference: *mut [Limb],
        source: *const Limb,
        mod_bits: usize,
        add_sub_kernel: unsafe fn(*mut Limb, *mut Limb, *const Limb, usize) -> (Limb, Limb),
    ) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: mod_bits/LIMB_BITS+1 <= usize::MAX/LIMB_BITS+1 fits usize.
        let cl = unsafe { ml.unchecked_add(1) };
        // SAFETY: ml < cl and the caller provides the complete coefficient spans.
        let sum_guard = unsafe { *sum.get_unchecked(ml) };
        // SAFETY: source covers cl limbs, so its guard is readable before the
        // permitted exact difference/source alias is overwritten by the kernel.
        let source_guard = unsafe { *source.add(ml) };
        debug_assert!(sum_guard <= 1, "a semi-normalized sum guard is at most one");
        debug_assert!(
            source_guard <= 1,
            "a semi-normalized source guard is at most one"
        );

        // SAFETY: the caller guarantees the ml-limb data spans and permitted exact
        // difference/source alias. Architecture selection, including any ADX
        // requirement, remains encapsulated in arch/.
        let (carry, borrow) =
            unsafe { add_sub_kernel(sum.as_mut_ptr(), difference.cast::<Limb>(), source, ml) };
        debug_assert!(
            carry <= 1 && borrow <= 1,
            "limb kernels return one-bit flags"
        );

        // If c = sum_guard + source_guard + carry, then 0 <= c <= 3. Choose
        // adjustment x = max(c-1, 0), set the guard to c-x in {0,1}, and subtract
        // x from the full coefficient. Since 2^n = -1, this changes the stored
        // integer by x*(2^n+1) and therefore preserves the ring value.
        // SAFETY: both input guards and carry are bits, so their sum is <=3.
        let sum_coefficient = unsafe { sum_guard.unchecked_add(source_guard).unchecked_add(carry) };
        let sum_adjustment = sum_coefficient.saturating_sub(1);
        // SAFETY: ml < cl and the caller guarantees sum has cl limbs.
        unsafe {
            *sum.get_unchecked_mut(ml) = Limb::from(sum_coefficient != 0);
        }
        // SAFETY: mod_bits is positive, so the data span contains limb zero.
        let (sum_low, sum_borrow) =
            unsafe { *sum.get_unchecked(0) }.overflowing_sub(sum_adjustment);
        // SAFETY: limb zero exists as proved above.
        unsafe {
            *sum.get_unchecked_mut(0) = sum_low;
        }
        if sum_borrow {
            // SAFETY: 1..cl is the remainder of the complete coefficient span.
            let _ = SsaCarry::propagate_borrow(unsafe { sum.get_unchecked_mut(1..cl) });
        }

        // SAFETY: after add_sub_kernel finishes, `source` is never accessed again in this function.
        // Converting raw slice pointer `difference` to `&mut [Limb]` exclusively borrows the buffer for post-kernel adjustments.
        let difference = unsafe { &mut *difference };

        // For the difference, c = sum_guard - source_guard - borrow lies in
        // {-2,-1,0,1}. Its signed high bit supplies the negative mask directly.
        // Adding -c to the full coefficient then leaves guard zero;
        // nonnegative c is already a valid semi-normalized guard.
        let difference_coefficient = sum_guard.wrapping_sub(source_guard).wrapping_sub(borrow);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss,
            reason = "The native-width two's-complement coefficient is -2, -1, 0, or 1; its arithmetic sign shift reinterprets exactly as a zero or all-ones limb mask on 16/32/64 bits"
        )]
        let negative_mask = ((difference_coefficient as isize) >> (Limb::BITS - 1)) as Limb;
        let difference_adjustment = difference_coefficient.wrapping_neg() & negative_mask;
        // SAFETY: ml < cl and the caller guarantees difference has cl limbs.
        unsafe {
            *difference.get_unchecked_mut(ml) =
                difference_coefficient.wrapping_add(difference_adjustment);
        }
        // SAFETY: mod_bits is positive, so difference contains limb zero.
        let (difference_low, difference_carry) =
            unsafe { *difference.get_unchecked(0) }.overflowing_add(difference_adjustment);
        // SAFETY: limb zero exists as proved above.
        unsafe {
            *difference.get_unchecked_mut(0) = difference_low;
        }
        if difference_carry {
            // SAFETY: 1..cl is the remainder of the complete coefficient span.
            let _ = SsaCarry::propagate_carry(unsafe { difference.get_unchecked_mut(1..cl) });
        }

        debug_assert!(
            // SAFETY: both guard indices are within their caller-provided spans.
            unsafe { *sum.get_unchecked(ml) } <= 1,
            "semi-normalized Fermat sum guard is at most one"
        );
        debug_assert!(
            // SAFETY: the same coefficient-width proof applies to difference.
            unsafe { *difference.get_unchecked(ml) } <= 1,
            "semi-normalized Fermat difference guard is at most one"
        );
    }

    /// Computes `dst = -dst mod (2^n + 1)` in-place.
    ///
    /// For `B = 2^n`, data value `L`, and guard `g`, the ring value is
    /// `L - g`. Complementing the data and adding `g + 2` forms
    /// `T = B + 1 + g - L`, which is congruent to its negation. The carry
    /// dependency ends at the first nonzero upper input limb; all remaining
    /// limbs require only independent complements.
    ///
    /// # Safety
    /// `mod_bits` is positive and limb-aligned. `dst` contains at least
    /// `coeff_limbs(mod_bits)` limbs and is semi-normalized (guard at most one).
    pub unsafe fn negate(dst: &mut [Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: 0 < ml < dst.len(); the split exclusively partitions the
        // initialized data from the guard and any unused destination suffix.
        let (data, high) = unsafe { dst.split_at_mut_unchecked(ml) };
        // SAFETY: the positive ring width gives at least one data limb.
        let (first, tail) = unsafe { data.split_first_mut().unwrap_unchecked() };
        // SAFETY: the complete coefficient includes high[0], its guard.
        let guard = unsafe { high.get_unchecked_mut(0) };
        debug_assert!(*guard <= 1, "a semi-normalized Fermat guard is at most one");
        // SAFETY: the semi-normalized guard is <=1, so guard+2<=3.
        let adjustment = unsafe { guard.unchecked_add(2) };
        let (low, mut carry) = (!*first).overflowing_add(adjustment);

        let mut remaining = tail.iter_mut();
        if carry {
            for limb in remaining.by_ref() {
                let (sum, escaped) = (!*limb).overflowing_add(1);
                *limb = sum;
                carry = escaped;
                if !carry {
                    break;
                }
            }
        }
        for limb in remaining {
            *limb = !*limb;
        }
        // 2 <= T <= B+2, so an escaped carry leaves data in {0,1,2}.
        // Zero data denotes B; otherwise subtract B+1 from this small value.
        let carry_digit = Limb::from(carry);
        let correction = carry_digit & Limb::from(low != 0);
        // SAFETY: correction is zero or one; a value of one requires low>0.
        *first = unsafe { low.unchecked_sub(correction) };
        *guard = carry_digit ^ correction;
    }

    /// Canonicalizes `dst` in `[0, 2^n]` and returns whether it represents `-1`.
    ///
    /// Reduces modulo $2^n + 1$ if the value has overflowed or underflowed.
    ///
    /// # Safety
    /// `mod_bits` is positive and limb-aligned; `dst.len() >= coeff_limbs(mod_bits)`.
    pub unsafe fn normalize(dst: &mut [Limb], mod_bits: usize) -> bool {
        let ml = Self::mod_limbs(mod_bits);
        // Writing the coefficient as guard*2^n + low and using 2^n = -1 shows
        // that its residue is exactly low - guard. Since guard is one limb and low
        // is nonnegative, one modulus correction is sufficient after underflow.
        // SAFETY: ml < coeff_limbs(mod_bits) <= dst.len().
        let guard = unsafe { *dst.get_unchecked(ml) };
        if guard == 0 {
            return false;
        }
        // SAFETY: ml < coeff_limbs(mod_bits) <= dst.len().
        unsafe {
            *dst.get_unchecked_mut(ml) = 0;
        }

        // SAFETY: mod_bits is a positive multiple of LIMB_BITS, so ml is nonzero.
        let (low, borrow) = unsafe { *dst.get_unchecked(0) }.overflowing_sub(guard);
        // SAFETY: mod_bits is a positive multiple of LIMB_BITS, so ml is nonzero.
        unsafe {
            *dst.get_unchecked_mut(0) = low;
        }
        // SAFETY: 1..ml is within the cl-limb coefficient.
        let escaped = borrow && SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(1..ml) });
        if escaped {
            // low - guard was negative. The wrapped subtraction already supplied
            // 2^n; add the remaining +1 from the modulus. Only a full carry is
            // represented by the guard value 2^n = -1.
            // SAFETY: caller guarantees dst has cl > ml limbs.
            return unsafe { SsaCarry::correct_wrapped_shift_difference(dst, ml) };
        }
        false
    }

    /// Computes `dst = dst + source` modulo `2^n + 1`, in place.
    ///
    /// The sum-only TFT butterfly does not write a discarded difference. For
    /// data carry c and input guards g,h, the combined guard g+h+c lies in
    /// 0..=3. Subtracting max(g+h+c-1,0) from the complete coefficient and
    /// retaining a guard of at most one preserves its value modulo `2^n+1`.
    ///
    /// # Safety
    /// Both disjoint slices contain a complete semi-normalized coefficient
    /// for the positive, limb-aligned ring width, with guards at most one.
    pub unsafe fn add_in_place(dst: &mut [Limb], source: &[Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: both complete coefficients include the guard at ml and at
        // least one data limb; the disjoint data spans have identical widths.
        let (left_guard, right_guard, carry) = unsafe {
            (
                *dst.get_unchecked(ml),
                *source.get_unchecked(ml),
                SsaCarry::add_full_in_place(
                    dst.get_unchecked_mut(..ml),
                    source.get_unchecked(..ml),
                ),
            )
        };
        debug_assert!(
            left_guard <= 1 && right_guard <= 1,
            "semi-normalized addition guards are at most one"
        );
        // SAFETY: input guards and the data carry are bits; their sum is <=3.
        let guard = unsafe { left_guard.unchecked_add(right_guard).unchecked_add(carry) };
        let adjustment = guard.saturating_sub(1);
        // SAFETY: limb zero and guard ml belong to the complete coefficient.
        // guard-adjustment <= 1; subtracting adjustment changes the integer
        // by adjustment*(2^n+1) relative to the uncorrected sum.
        let borrow = unsafe {
            *dst.get_unchecked_mut(ml) = Limb::from(guard != 0);
            let (low, escaped) = (*dst.get_unchecked(0)).overflowing_sub(adjustment);
            *dst.get_unchecked_mut(0) = low;
            escaped
        };
        if borrow {
            // SAFETY: dst has ml+1 limbs, and the corrected positive guard
            // absorbs the propagated borrow before it escapes the coefficient.
            let _ = SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(1..=ml) });
        }
    }

    /// Computes `dst = dst - source` modulo `2^n + 1`, in place.
    ///
    /// This is the difference half of [`Self::add_sub`] on its own, for callers
    /// that have no use for the sum and would otherwise pay for a second output.
    /// Inputs and outputs are semi-normalized on the same terms.
    ///
    /// # Safety
    /// - Both slices cover at least `coeff_limbs(mod_bits)` limbs.
    /// - `source` is disjoint from `dst`.
    pub unsafe fn sub_in_place(dst: &mut [Limb], source: &[Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // SAFETY: ml<=usize::MAX/LIMB_BITS, so the additional guard fits.
        let cl = unsafe { ml.unchecked_add(1) };
        // SAFETY: ml < cl and the caller provides complete coefficient spans.
        let dst_guard = unsafe { *dst.get_unchecked(ml) };
        // SAFETY: same span guarantee as `dst`.
        let source_guard = unsafe { *source.get_unchecked(ml) };
        debug_assert!(
            dst_guard <= 1 && source_guard <= 1,
            "semi-normalized guards are at most one"
        );

        // SAFETY: both spans contain at least ml limbs.
        let borrow = unsafe {
            SsaCarry::sub_full_in_place(dst.get_unchecked_mut(..ml), source.get_unchecked(..ml))
        };

        // `c = dst_guard - source_guard - borrow` lies in {-2,-1,0,1}. Its wrapping
        // signed high bit supplies its negative mask; adding `-c` to the full
        // coefficient then leaves a guard of zero. This is the same signed
        // reduction `Self::add_sub` applies to its difference output.
        let coefficient = dst_guard.wrapping_sub(source_guard).wrapping_sub(borrow);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss,
            reason = "The native-width two's-complement coefficient is -2, -1, 0, or 1; its arithmetic sign shift reinterprets exactly as a zero or all-ones limb mask on 16/32/64 bits"
        )]
        let negative_mask = ((coefficient as isize) >> (Limb::BITS - 1)) as Limb;
        let adjustment = coefficient.wrapping_neg() & negative_mask;
        // SAFETY: ml < cl and the caller guarantees dst has cl limbs.
        unsafe {
            *dst.get_unchecked_mut(ml) = coefficient.wrapping_add(adjustment);
        }
        // SAFETY: mod_bits is positive, so the data span contains limb zero.
        let (low, carry) = unsafe { *dst.get_unchecked(0) }.overflowing_add(adjustment);
        // SAFETY: limb zero exists as proved above.
        unsafe {
            *dst.get_unchecked_mut(0) = low;
        }
        if carry {
            // SAFETY: 1..cl is the remainder of the complete coefficient span.
            let _ = SsaCarry::propagate_carry(unsafe { dst.get_unchecked_mut(1..cl) });
        }
    }

    /// Number of data limbs in a Fermat ring element, excluding the guard limb.
    #[expect(
        clippy::inline_always,
        reason = "single-instruction division constant folded on every call site"
    )]
    #[inline(always)]
    pub const fn mod_limbs(mod_bits: usize) -> usize {
        mod_bits.div_euclid(LIMB_BITS)
    }

    /// Total slot width for a ring element: [`Self::mod_limbs`] plus the guard limb that
    /// accommodates the value $2^n$, which needs `n + 1` significant bits.
    #[expect(
        clippy::inline_always,
        reason = "constant-folded slot width used on every coefficient access"
    )]
    #[inline(always)]
    pub const fn coeff_limbs(mod_bits: usize) -> NonZeroUsize {
        // SAFETY: LIMB_BITS>=16 bounds the quotient by usize::MAX/16;
        // adding its guard cannot overflow and makes the slot positive on
        // every pointer width, including a zero-data sizing request.
        unsafe { NonZeroUsize::new_unchecked(mod_bits.div_euclid(LIMB_BITS).unchecked_add(1)) }
    }

    /// Computes `dst = 0 - src mod B^N` directly without pre-zeroing `dst`.
    ///
    /// Both slices have equal length. Returns whether subtraction borrowed,
    /// which happens exactly when `src` contains at least one nonzero limb.
    /// Before that limb the result is zero. At it the result is its modular
    /// negation, and the borrow then remains one: every remaining digit is
    /// `0 - source - 1 == !source`, independently of the preceding digit.
    #[inline]
    pub fn neg_slice_into(dst: &mut [Limb], src: &[Limb]) -> bool {
        debug_assert_eq!(
            dst.len(),
            src.len(),
            "negation spans must cover the same limb count"
        );
        let mut digits = dst.iter_mut().zip(src);
        let mut borrowed = false;
        for (destination, &source) in digits.by_ref() {
            *destination = source.wrapping_neg();
            if source != 0 {
                borrowed = true;
                break;
            }
        }
        for (destination, &source) in digits {
            *destination = !source;
        }
        borrowed
    }

    /// Reduces a shift amount modulo the ring's full period `2 * mod_bits`.
    ///
    /// Two is a `2 * mod_bits`-th root of unity in this ring, so every shift is
    /// meaningful only modulo that period. The period is a power of two whenever
    /// the ring width is, which is the common case and reduces to a mask; the
    /// remainder path covers the alignment-derived widths the planner also emits.
    /// The plan establishes positivity once and carries it in `period`.
    #[inline]
    pub fn reduce_mod_period(x: usize, period: NonZeroUsize) -> usize {
        if period.is_power_of_two() {
            // SAFETY: NonZeroUsize carries the positive period from construction.
            x & unsafe { period.get().unchecked_sub(1) }
        } else {
            x % period
        }
    }
}
