//! Exact division and halving of fixed-width values.
//!
//! Every routine here divides a value that is *known* to be divisible, so each
//! is a single low-to-high recurrence rather than a general division. For an odd
//! divisor the B-adic quotient is unique modulo the fixed width, which is what
//! makes these exact on two's-complement negative intermediates as well.

use super::{ArchKernels, LIMB_BITS, Limb, SharedEval};

const HIGH_BIT: Limb = Limb::MAX ^ (Limb::MAX >> 1);

// ── Power-of-two division ────────────────────────────────────────────────────

impl SharedEval {
    /// Divide a fixed-width value exactly by two.
    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
    #[inline(always)]
    pub fn exact_div2_in_place(value: &mut [Limb]) {
        if value.is_empty() {
            return;
        }
        // SAFETY: value is non-empty and 0 < 1 < LIMB_BITS on every supported target.
        // Exact divisibility proves the discarded low bit is zero.
        unsafe {
            let _ = ArchKernels::rshift_unchecked(value.as_mut_ptr(), value.len(), 1);
        }
    }

    /// Divide a fixed-width value exactly by four in one right-shift pass.
    pub fn exact_div4_in_place(value: &mut [Limb]) {
        if value.is_empty() {
            return;
        }
        // SAFETY: value is non-empty and 0 < 2 < LIMB_BITS on every supported
        // target. Exact divisibility proves both discarded low bits are zero.
        unsafe {
            let _ = ArchKernels::rshift_unchecked(value.as_mut_ptr(), value.len(), 2);
        }
    }

    /// Divide a fixed-width value exactly by `2^shift` in one right-shift pass.
    pub fn exact_div_power_of_two_in_place(value: &mut [Limb], shift: u32) {
        if shift == 0 || value.is_empty() {
            return;
        }
        debug_assert!(
            shift < Limb::BITS,
            "exact power-of-two division shift exceeds one limb"
        );
        let low_mask = Limb::MAX.wrapping_shr(Limb::BITS.wrapping_sub(shift));
        debug_assert_eq!(
            value.first().copied().unwrap_or(0) & low_mask,
            0,
            "exact power-of-two division discarded nonzero low bits"
        );
        // SAFETY: value is non-empty and 0 < shift < LIMB_BITS. Exact divisibility
        // proves every discarded low bit is zero.
        unsafe {
            let _ = ArchKernels::rshift_unchecked(value.as_mut_ptr(), value.len(), shift);
        }
    }

    /// Divide a fixed-width two's-complement value exactly by `2^shift`.
    pub fn exact_signed_div_power_of_two_in_place(value: &mut [Limb], shift: u32) {
        if shift == 0 || value.is_empty() {
            return;
        }
        debug_assert!(shift < Limb::BITS, "signed shift exceeds one limb");
        let inverse_shift = Limb::BITS.wrapping_sub(shift);
        let low_mask = Limb::MAX.wrapping_shr(inverse_shift);
        debug_assert_eq!(
            value.first().copied().unwrap_or(0) & low_mask,
            0,
            "exact signed division discarded nonzero low bits"
        );
        let sign_extension = Limb::from(value.last().copied().unwrap_or(0) & HIGH_BIT != 0)
            .wrapping_mul(Limb::MAX.wrapping_shl(inverse_shift));
        let mut incoming = sign_extension;
        for limb in value.iter_mut().rev() {
            let next = limb.wrapping_shl(inverse_shift);
            *limb = limb.wrapping_shr(shift) | incoming;
            incoming = next;
        }
    }

    // ── Combine-and-halve ────────────────────────────────────────────────────────

    /// Replace `value` with the exact half of `value + positive`.
    ///
    /// The sum is treated as one bit wider than the buffer: an escaping carry
    /// becomes the quotient's top bit.
    pub fn exact_half_sum_in_place(value: &mut [Limb], positive: &[Limb]) {
        Self::exact_half_combined_in_place::<false, true>(value, positive);
    }

    /// Replace `value` with the exact half of `(value + other) mod B^n`.
    ///
    /// Unlike [`Self::exact_half_sum_in_place`], the final carry is intentionally
    /// discarded before halving. This is the signed fixed-width operation needed
    /// when `value` is a two's-complement interpolation difference but the modular
    /// sum is a proven nonnegative even coefficient.
    pub fn exact_half_modular_sum_in_place(value: &mut [Limb], other: &[Limb]) {
        Self::exact_half_combined_in_place::<false, false>(value, other);
    }

    /// Replace `value` with the exact half of `positive - value`.
    ///
    /// The difference is proven nonnegative, so no borrow escapes and there is no
    /// carry to place.
    pub fn exact_half_reverse_difference_in_place(value: &mut [Limb], positive: &[Limb]) {
        Self::exact_half_combined_in_place::<true, false>(value, positive);
    }

    // ── Odd-divisor division ─────────────────────────────────────────────────────

    /// Return the multiplicative inverse of an odd limb modulo `2^LIMB_BITS`.
    pub const fn invert_odd(divisor: Limb) -> Limb {
        let mut inverse = 1_usize;
        let mut correct_bits = 1_usize;
        while correct_bits < LIMB_BITS {
            inverse = inverse.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inverse)));
            correct_bits = correct_bits.wrapping_mul(2);
        }
        inverse
    }

    /// Divide a fixed-width two's-complement value exactly by an odd limb.
    pub fn exact_div_odd_in_place(value: &mut [Limb], divisor: Limb, inverse: Limb) {
        debug_assert!(
            divisor != 0 && divisor & 1 == 1,
            "exact fixed-width division requires a nonzero odd divisor"
        );
        let mut borrow = 0;
        for limb in value {
            let (adjusted, underflow) = limb.overflowing_sub(borrow);
            let quotient = adjusted.wrapping_mul(inverse);
            let (_, high) = ArchKernels::mul_limb_lo_hi(quotient, divisor);
            borrow = high.wrapping_add(Limb::from(underflow));
            *limb = quotient;
        }
    }

    /// Divide a fixed-width two's-complement value exactly by a divisor of `B-1`.
    ///
    /// Every supported limb width is a multiple of eight, so `3`, `15`, and `255`
    /// all divide `B-1`. Multiplying each input limb by `(B-1)/DIVISOR` turns exact
    /// division into a low-to-high radix-minus-one recurrence: if
    /// `p = limb*(B-1)/DIVISOR`, the next quotient limb is `high - p.low`, and
    /// subtracting `p.high` and that subtraction's borrow gives the state for the
    /// next radix position. This needs one full multiplication per limb, rather
    /// than the modular-inverse multiplication plus a second multiplication to
    /// recover the carry that the general odd-divisor recurrence above requires.
    ///
    /// Since every such divisor is odd, the B-adic quotient is unique modulo the
    /// fixed width, including for two's-complement negative intermediates.
    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
    #[inline(always)]
    pub fn exact_div_radix_minus_one_in_place<const DIVISOR: Limb>(value: &mut [Limb]) {
        const {
            assert!(
                DIVISOR & 1 == 1 && Limb::MAX.wrapping_rem(DIVISOR) == 0,
                "the radix-minus-one recurrence requires an odd divisor of B-1"
            );
        }
        let factor = Limb::MAX.div_euclid(DIVISOR);

        let mut high = 0;
        for limb in value {
            let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(*limb, factor);
            let low_borrow = Limb::from(high < product_low);
            let quotient = high.wrapping_sub(product_low);
            *limb = quotient;
            high = quotient.wrapping_sub(product_high).wrapping_sub(low_borrow);
        }
    }

    /// Divide a fixed-width two's-complement value exactly by nine.
    ///
    /// Nine does not divide `B-1`, so this uses the modular inverse. For
    /// `9*q = (q<<3)+q`, the high product limb is `q>>(w-3)` plus the carry from
    /// adding `q` to the shifted low limb, which replaces the second full-width
    /// multiplication of [`Self::exact_div_odd_in_place`] with two shifts.
    pub fn exact_div9_in_place(value: &mut [Limb]) {
        const INVERSE: Limb = SharedEval::invert_odd(9);
        const HIGH_SHIFT: u32 = Limb::BITS.wrapping_sub(3);

        let mut borrow = 0;
        for limb in value {
            let (adjusted, underflow) = limb.overflowing_sub(borrow);
            let quotient = adjusted.wrapping_mul(INVERSE);
            let shifted_low = quotient.wrapping_shl(3);
            let (_, low_carry) = shifted_low.overflowing_add(quotient);
            let product_high = quotient
                .wrapping_shr(HIGH_SHIFT)
                .wrapping_add(Limb::from(low_carry));
            borrow = product_high.wrapping_add(Limb::from(underflow));
            *limb = quotient;
        }
    }

    /// Replace `dst` with the exact quotient `(dst - scalar * src) / divisor`.
    ///
    /// The subtraction and odd exact division both propagate from low to high,
    /// so carrying both recurrences in one pass preserves their radix-`B`
    /// invariants while avoiding an intermediate full-buffer write and reread.
    pub fn exact_sub_mul_word_odd_in_place(
        dst: &mut [Limb],
        src: &[Limb],
        scalar: Limb,
        divisor: Limb,
    ) {
        debug_assert_eq!(dst.len(), src.len(), "fused interpolation widths differ");
        debug_assert!(divisor & 1 == 1, "exact divisor must be odd");
        let inverse = Self::invert_odd(divisor);
        let mut product_carry = 0;
        let mut division_borrow = 0;
        for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
            let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(*src_limb, scalar);
            let (low_with_carry, carry_overflow) = product_low.overflowing_add(product_carry);
            let (difference, subtraction_underflow) = dst_limb.overflowing_sub(low_with_carry);
            product_carry = product_high
                .wrapping_add(Limb::from(carry_overflow))
                .wrapping_add(Limb::from(subtraction_underflow));

            let (adjusted, division_underflow) = difference.overflowing_sub(division_borrow);
            let quotient = adjusted.wrapping_mul(inverse);
            let (_, quotient_high) = ArchKernels::mul_limb_lo_hi(quotient, divisor);
            division_borrow = quotient_high.wrapping_add(Limb::from(division_underflow));
            *dst_limb = quotient;
        }
        // Arithmetic is modulo B^n. Exact divisibility proves the quotient limbs;
        // final carries are only discarded sign extension beyond the guard.
        let _ = (product_carry, division_borrow);
    }

    /// One combining pass that halves `value ± other` as it goes.
    ///
    /// The three published forms differ only in how the pair is combined and what
    /// becomes of the final carry, so they share this driver and instantiate it;
    /// the loop, its carry chain, and its exactness argument exist once.
    ///
    /// `REVERSE` computes `other - value` instead of `value + other`.
    /// `SIGN_EXTEND` folds the escaping carry into the top bit rather than
    /// discarding it, which is the difference between a widening sum and one taken
    /// modulo `B^n`.
    #[allow(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
    #[inline(always)]
    fn exact_half_combined_in_place<const REVERSE: bool, const SIGN_EXTEND: bool>(
        value: &mut [Limb],
        other: &[Limb],
    ) {
        debug_assert_eq!(value.len(), other.len(), "exact-half widths must match");
        let mut pairs = value.iter_mut().zip(other);
        let Some((first_dst, first_src)) = pairs.next() else {
            return;
        };
        let (first_combined, first_overflow) = if REVERSE {
            first_src.overflowing_sub(*first_dst)
        } else {
            first_dst.overflowing_add(*first_src)
        };
        debug_assert_eq!(first_combined & 1, 0, "exact half discarded a nonzero bit");
        let mut previous_dst = first_dst;
        let mut previous_value = first_combined;
        let mut carry = Limb::from(first_overflow);

        for (current_dst, current_src) in pairs {
            let (combined, overflow_a) = if REVERSE {
                current_src.overflowing_sub(*current_dst)
            } else {
                current_dst.overflowing_add(*current_src)
            };
            let (current_value, overflow_b) = if REVERSE {
                combined.overflowing_sub(carry)
            } else {
                combined.overflowing_add(carry)
            };
            *previous_dst =
                (previous_value >> 1) | Limb::from(current_value & 1 != 0).wrapping_mul(HIGH_BIT);
            previous_dst = current_dst;
            previous_value = current_value;
            carry = Limb::from(overflow_a | overflow_b);
        }
        if REVERSE {
            debug_assert_eq!(carry, 0, "reverse difference became negative");
        }
        *previous_dst = if SIGN_EXTEND {
            (previous_value >> 1) | Limb::from(carry & 1 != 0).wrapping_mul(HIGH_BIT)
        } else {
            previous_value >> 1
        };
    }
}
