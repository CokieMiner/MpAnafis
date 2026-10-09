//! Exact division and halving of fixed-width values.
//!
//! Divisibility is established by interpolation. Odd divisors have a unique
//! quotient modulo the fixed width, including for two's-complement values.

#![expect(
    unsafe_code,
    reason = "Exact divisibility, bounded shifts, and scalar carry limits prove the fixed-width recurrences"
)]

use super::{ArchKernels, LIMB_BITS, Limb, SharedEval};

const HIGH_BIT: Limb = Limb::MAX ^ (Limb::MAX >> 1);

impl SharedEval {
    /// Divide a fixed-width value exactly by two.
    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
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
        // SAFETY: the early return and shift contract give 0 < shift < Limb::BITS.
        let inverse_shift = unsafe { Limb::BITS.unchecked_sub(shift) };
        let low_mask = Limb::MAX >> inverse_shift;
        // SAFETY: the empty slice case returned above.
        let low = unsafe { *value.get_unchecked(0) };
        debug_assert_eq!(
            low & low_mask,
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
        // SAFETY: the early return and shift contract give 0 < shift < Limb::BITS.
        let inverse_shift = unsafe { Limb::BITS.unchecked_sub(shift) };
        let low_mask = Limb::MAX >> inverse_shift;
        // SAFETY: the empty slice case returned above, so both end indices exist.
        let (low, high) = unsafe {
            (
                *value.get_unchecked(0),
                *value.get_unchecked(value.len().unchecked_sub(1)),
            )
        };
        debug_assert_eq!(
            low & low_mask,
            0,
            "exact signed division discarded nonzero low bits"
        );
        // SAFETY: the sign factor is binary, so multiplication by the shifted
        // all-ones limb produces either zero or that limb without overflow.
        let sign_extension =
            unsafe { Limb::from(high & HIGH_BIT != 0).unchecked_mul(Limb::MAX << inverse_shift) };
        let mut incoming = sign_extension;
        for limb in value.iter_mut().rev() {
            let next = *limb << inverse_shift;
            *limb = (*limb >> shift) | incoming;
            incoming = next;
        }
    }

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

    /// Return the multiplicative inverse of an odd limb modulo `2^LIMB_BITS`.
    pub const fn invert_odd(divisor: Limb) -> Limb {
        let mut inverse = 1_usize;
        let mut correct_bits = 1_usize;
        while correct_bits < LIMB_BITS {
            inverse = inverse.wrapping_mul(2_usize.wrapping_sub(divisor.wrapping_mul(inverse)));
            // SAFETY: LIMB_BITS is 16, 32, or 64. Starting at one and doubling
            // below this power of two never exceeds LIMB_BITS.
            correct_bits = unsafe { correct_bits.unchecked_mul(2) };
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
            // SAFETY: quotient < B and 1 <= divisor < B give high <= divisor-1;
            // adding the binary borrow is at most divisor <= Limb::MAX.
            borrow = unsafe { high.unchecked_add(Limb::from(underflow)) };
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
    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
    #[inline(always)]
    pub fn exact_div_radix_minus_one_in_place<const DIVISOR: Limb>(value: &mut [Limb]) {
        const {
            assert!(
                DIVISOR & 1 == 1 && Limb::MAX.rem_euclid(DIVISOR) == 0,
                "the radix-minus-one recurrence requires an odd divisor of B-1"
            );
        }
        let factor = Limb::MAX.div_euclid(DIVISOR);

        let mut high = 0;
        let (chunks, remainder) = value.as_chunks_mut::<4>();
        for chunk in chunks {
            let [l0, l1, l2, l3] = *chunk;
            let (pl0, ph0) = ArchKernels::mul_limb_lo_hi(l0, factor);
            let (pl1, ph1) = ArchKernels::mul_limb_lo_hi(l1, factor);
            let (pl2, ph2) = ArchKernels::mul_limb_lo_hi(l2, factor);
            let (pl3, ph3) = ArchKernels::mul_limb_lo_hi(l3, factor);

            let low_borrow0 = Limb::from(high < pl0);
            let q0 = high.wrapping_sub(pl0);
            let h1 = q0.wrapping_sub(ph0).wrapping_sub(low_borrow0);

            let low_borrow1 = Limb::from(h1 < pl1);
            let q1 = h1.wrapping_sub(pl1);
            let h2 = q1.wrapping_sub(ph1).wrapping_sub(low_borrow1);

            let low_borrow2 = Limb::from(h2 < pl2);
            let q2 = h2.wrapping_sub(pl2);
            let h3 = q2.wrapping_sub(ph2).wrapping_sub(low_borrow2);

            let low_borrow3 = Limb::from(h3 < pl3);
            let q3 = h3.wrapping_sub(pl3);
            high = q3.wrapping_sub(ph3).wrapping_sub(low_borrow3);

            *chunk = [q0, q1, q2, q3];
        }

        for limb in remainder {
            let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(*limb, factor);
            let low_borrow = Limb::from(high < product_low);
            let quotient = high.wrapping_sub(product_low);
            *limb = quotient;
            high = quotient.wrapping_sub(product_high).wrapping_sub(low_borrow);
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
            // SAFETY: inductively product_carry <= scalar. Thus
            // src_limb*scalar+product_carry <= B*scalar. Its high limb is at
            // most scalar; equality forces low_with_carry == 0, which cannot
            // borrow from dst_limb. Otherwise the binary borrow raises the high
            // limb only to scalar. Both additions therefore fit Limb, even for
            // scalar == B-1; scalar == 0 retains zero carry throughout.
            product_carry = unsafe {
                product_high
                    .unchecked_add(Limb::from(carry_overflow))
                    .unchecked_add(Limb::from(subtraction_underflow))
            };

            let (adjusted, division_underflow) = difference.overflowing_sub(division_borrow);
            let quotient = adjusted.wrapping_mul(inverse);
            let (_, quotient_high) = ArchKernels::mul_limb_lo_hi(quotient, divisor);
            // SAFETY: quotient < B and the positive odd divisor < B give
            // quotient_high <= divisor-1; its binary borrow raises it at most
            // to divisor <= Limb::MAX.
            division_borrow =
                unsafe { quotient_high.unchecked_add(Limb::from(division_underflow)) };
            *dst_limb = quotient;
        }
        // Arithmetic is modulo B^n. Exact divisibility proves the quotient limbs;
        // final carries are only discarded sign extension beyond the guard.
        let _ = (product_carry, division_borrow);
    }

    /// One combining pass that halves `value +/- other` as it goes.
    ///
    /// `REVERSE` computes `other - value` instead of `value + other`.
    /// `SIGN_EXTEND` folds the escaping carry into the top bit rather than
    /// discarding it, which is the difference between a widening sum and one taken
    /// modulo `B^n`.
    #[expect(clippy::inline_always, reason = "Critical for Toom-Cook interpolation")]
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
            *previous_dst = (previous_value >> 1) | ((current_value & 1) << (LIMB_BITS - 1));
            previous_dst = current_dst;
            previous_value = current_value;
            carry = Limb::from(overflow_a | overflow_b);
        }
        if REVERSE {
            debug_assert_eq!(carry, 0, "reverse difference became negative");
        }
        *previous_dst = if SIGN_EXTEND {
            (previous_value >> 1) | (carry << (LIMB_BITS - 1))
        } else {
            previous_value >> 1
        };
    }
}
