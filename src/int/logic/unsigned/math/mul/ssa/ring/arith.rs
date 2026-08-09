use super::{LIMB_BITS, Limb, SsaCarry};

/// A canonical Fermat residue's relation to the two multiplicative special
/// cases that every product path short-circuits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Residue {
    /// The additive identity: every data limb and the guard limb are zero.
    Zero,
    /// The residue `-1 ≡ 2^n`: every data limb is zero and the guard is one.
    NegOne,
    /// Anything else, including a non-canonical guard above one.
    Ordinary,
}

/// Namespace for arithmetic in the Fermat ring `Z/(2^mod_bits + 1)`.
///
/// The whole `ring` folder contributes to this one namespace: [`arith`](self)
/// supplies the slot widths and the add/subtract/negate/normalize family,
/// [`shift`](super::shift) the in-place multiplications by a power of two, and
/// [`shift_from`](super::shift_from) their out-of-place form. Every method takes
/// `mod_bits` explicitly rather than the namespace carrying it, because one
/// transform nests rings of several widths at once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsaRing;

impl SsaRing {
    /// Classifies a canonical residue with a single pass over its data limbs.
    ///
    /// Both special cases require the data limbs to be all-zero and differ only in
    /// the guard, so testing them separately sweeps the coefficient twice for a
    /// fact that one scan already settles. The pointwise stage runs this on every
    /// coefficient of both matrices, where the operands are almost never special,
    /// so the redundant sweep is pure loss.
    ///
    /// # Safety
    /// `coeff.len() >= coeff_limbs(mod_bits)`, i.e. at least `ml + 1` limbs.
    #[allow(
        clippy::inline_always,
        reason = "single-pass classification on the hot pointwise path"
    )]
    #[inline(always)]
    pub unsafe fn classify_residue(coeff: &[Limb], ml: usize) -> Residue {
        // SAFETY: caller guarantees coeff.len() > ml, so both accesses are in range.
        let data_zero = unsafe { coeff.get_unchecked(..ml) }.iter().all(|l| *l == 0);
        if !data_zero {
            return Residue::Ordinary;
        }
        // SAFETY: ml < coeff.len() as guaranteed by the caller.
        match unsafe { *coeff.get_unchecked(ml) } {
            0 => Residue::Zero,
            1 => Residue::NegOne,
            _ => Residue::Ordinary,
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
        let cl = ml.wrapping_add(1);
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
            unsafe { add_sub_kernel(sum.as_mut_ptr(), (*difference).as_mut_ptr(), source, ml) };
        debug_assert!(
            carry <= 1 && borrow <= 1,
            "limb kernels return one-bit flags"
        );

        // If c = sum_guard + source_guard + carry, then 0 <= c <= 3. Choose
        // adjustment x = max(c-1, 0), set the guard to c-x in {0,1}, and subtract
        // x from the full coefficient. Since 2^n = -1, this changes the stored
        // integer by x*(2^n+1) and therefore preserves the ring value.
        let sum_coefficient = sum_guard.wrapping_add(source_guard).wrapping_add(carry);
        let nonzero_mask = 0_usize.wrapping_sub(Limb::from(sum_coefficient != 0));
        let sum_adjustment = sum_coefficient.wrapping_sub(1) & nonzero_mask;
        // SAFETY: ml < cl and the caller guarantees sum has cl limbs.
        unsafe {
            *sum.get_unchecked_mut(ml) = sum_coefficient.wrapping_sub(sum_adjustment);
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
        // {-2,-1,0,1}. Its wrapping representation exceeds one exactly when c is
        // negative. Adding -c to the full coefficient then leaves guard zero;
        // nonnegative c is already a valid semi-normalized guard.
        let difference_coefficient = sum_guard.wrapping_sub(source_guard).wrapping_sub(borrow);
        let negative_mask = 0_usize.wrapping_sub(Limb::from(difference_coefficient > 1));
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
    /// For zero, this is a no-op. For nonzero `v`, computes `2^n + 1 - v`.
    ///
    /// # Safety
    /// `dst.len() >= coeff_limbs(mod_bits)`.
    pub unsafe fn negate(dst: &mut [Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        let cl = ml.wrapping_add(1);

        // SAFETY: caller guarantees dst.len() >= cl.
        if unsafe { dst.get_unchecked(..cl) }.iter().all(|l| *l == 0) {
            return;
        }

        // Step 1: Negate all data limbs (two's complement of the full cl-limb value).
        for i in 0..cl {
            // SAFETY: i < cl, in bounds.
            let limb_ref = unsafe { dst.get_unchecked_mut(i) };
            *limb_ref = !*limb_ref;
        }
        // Add 2^n + 2: guard limb += 1, limb[0] += 2.
        // SAFETY: 0 < cl, in bounds.
        let (sum_lo, carry_lo) = unsafe { dst.get_unchecked(0) }.overflowing_add(2);
        // SAFETY: 0 < cl, in bounds.
        unsafe {
            *dst.get_unchecked_mut(0) = sum_lo;
        }
        if carry_lo {
            // SAFETY: 1..cl is within dst when cl > 1.
            let _ = SsaCarry::propagate_carry(unsafe { dst.get_unchecked_mut(1..cl) });
        }
        // SAFETY: ml < cl, in bounds.
        let guard_ref = unsafe { dst.get_unchecked_mut(ml) };
        *guard_ref = guard_ref.wrapping_add(1);

        // If result overflowed the modulus; subtract 2^n + 1.
        // SAFETY: ml < cl, in bounds.
        let guard_val = unsafe { *dst.get_unchecked(ml) };
        if guard_val > 1 {
            // SAFETY: ml < cl, in bounds.
            unsafe {
                *dst.get_unchecked_mut(ml) = guard_val.wrapping_sub(1);
            }
            // SAFETY: 0 < cl, in bounds.
            let (diff, borrow) = unsafe { dst.get_unchecked(0) }.overflowing_sub(1);
            // SAFETY: 0 < cl, in bounds.
            unsafe {
                *dst.get_unchecked_mut(0) = diff;
            }
            if borrow {
                // SAFETY: 1..ml is within dst when ml > 1.
                let _ = SsaCarry::propagate_borrow(unsafe { dst.get_unchecked_mut(1..ml) });
            }
        }
    }

    /// Ensures `dst` is a canonical residue in `[0, 2^n]`.
    ///
    /// Reduces modulo $2^n + 1$ if the value has overflowed or underflowed.
    ///
    /// # Safety
    /// `dst.len() >= coeff_limbs(mod_bits)`.
    pub unsafe fn normalize(dst: &mut [Limb], mod_bits: usize) {
        let ml = Self::mod_limbs(mod_bits);
        // Writing the coefficient as guard*2^n + low and using 2^n = -1 shows
        // that its residue is exactly low - guard. Since guard is one limb and low
        // is nonnegative, one modulus correction is sufficient after underflow.
        // SAFETY: ml < coeff_limbs(mod_bits) <= dst.len().
        let guard = unsafe { *dst.get_unchecked(ml) };
        if guard == 0 {
            return;
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
            unsafe {
                SsaCarry::correct_wrapped_shift_difference(dst, ml);
            }
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
        let cl = ml.wrapping_add(1);
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
        // representation exceeds one exactly when it is negative; adding `-c` to the
        // full coefficient then leaves a guard of zero. This is the same signed
        // reduction `Self::add_sub` applies to its difference output.
        let coefficient = dst_guard.wrapping_sub(source_guard).wrapping_sub(borrow);
        let negative_mask = 0_usize.wrapping_sub(Limb::from(coefficient > 1));
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

    // ── Slot widths and shift period ──────────────────────────────────────────

    /// Number of data limbs in a Fermat ring element, excluding the guard limb.
    #[allow(
        clippy::inline_always,
        reason = "single-instruction division constant folded on every call site"
    )]
    #[inline(always)]
    pub const fn mod_limbs(mod_bits: usize) -> usize {
        mod_bits.div_euclid(LIMB_BITS)
    }

    /// Total slot width for a ring element: [`Self::mod_limbs`] plus the guard limb that
    /// accommodates the value $2^n$, which needs `n + 1` significant bits.
    #[allow(
        clippy::inline_always,
        reason = "constant-folded slot width used on every coefficient access"
    )]
    #[inline(always)]
    pub const fn coeff_limbs(mod_bits: usize) -> usize {
        mod_bits.div_euclid(LIMB_BITS).wrapping_add(1)
    }

    /// Reduces a shift amount modulo the ring's full period `2 * mod_bits`.
    ///
    /// Two is a `2 * mod_bits`-th root of unity in this ring, so every shift is
    /// meaningful only modulo that period. The period is a power of two whenever
    /// the ring width is, which is the common case and reduces to a mask; the
    /// remainder path covers the alignment-derived widths the planner also emits.
    #[inline]
    pub const fn reduce_mod_period(x: usize, period: usize) -> usize {
        if period.is_power_of_two() {
            x & (period.wrapping_sub(1))
        } else {
            match x.checked_rem(period) {
                Some(remainder) => remainder,
                // Unreachable for a non-zero period, which every caller derives
                // from a positive ring width.
                None => 0,
            }
        }
    }
}
