//! Shared normalized two-half-limb division for targets with native limb division.
//!
//! Reference: D. E. Knuth, *The Art of Computer Programming*, Vol. 2,
//! Section 4.3.1, Algorithm D (3rd edition, 1997).

use core::num::NonZeroUsize;

use super::Limb;

/// Divide a two-limb numerator using two native limb-by-half-limb divisions.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor`.
#[inline(always)]
pub const unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    const HALF_BITS: u32 = Limb::BITS >> 1;
    const HALF_BASE: Limb = 1 << HALF_BITS;
    const HALF_MASK: Limb = HALF_BASE - 1;

    // SAFETY: the kernel contract requires divisor > 0 on every path.
    let denominator = unsafe { NonZeroUsize::new_unchecked(divisor) };

    // A scalar numerator needs one native divide, without normalization or
    // half-limb quotient corrections. Constant-zero callers erase this test.
    if rem_hi == 0 {
        // SAFETY: the caller guarantees divisor > 0, so the scalar quotient
        // fits Limb and quotient*divisor <= limb cannot overflow.
        unsafe {
            let quotient = limb.checked_div(denominator.get()).unwrap_unchecked();
            return (
                quotient,
                limb.unchecked_sub(quotient.unchecked_mul(denominator.get())),
            );
        }
    }

    let normalization = denominator.leading_zeros();
    let divisor_norm = denominator.get() << normalization;
    // SAFETY: normalization puts the high half in [HALF_BASE/2, HALF_BASE).
    let divisor_high = unsafe { NonZeroUsize::new_unchecked(divisor_norm >> HALF_BITS) };
    let divisor_low = divisor_norm & HALF_MASK;
    let numerator_high = if normalization == 0 {
        rem_hi
    } else {
        // SAFETY: denominator is nonzero, so 0<normalization<Limb::BITS
        // in this branch and the complementary shift is in the same range.
        let complement = unsafe { Limb::BITS.unchecked_sub(normalization) };
        (rem_hi << normalization) | (limb >> complement)
    };
    let numerator_low = limb << normalization;
    let numerator_mid = numerator_low >> HALF_BITS;
    let numerator_bottom = numerator_low & HALF_MASK;

    // With H=HALF_BASE, D=d1*H+d0 is normalized, so d1>=H/2.
    // N<D bounds each trial quotient by H+1 and its correction by two.
    // SAFETY: divisor_high stores the nonzero normalized high half.
    let mut quotient_high = unsafe {
        numerator_high
            .checked_div(divisor_high.get())
            .unwrap_unchecked()
    };
    // SAFETY: q=floor(N/d1) proves q*d1<=N; both operations are exact.
    let trial_remainder =
        unsafe { numerator_high.unchecked_sub(quotient_high.unchecked_mul(divisor_high.get())) };
    // SAFETY: q<=H+1 and d0<H bound q*d0 by H^2-1, one limb.
    let high_product = unsafe { quotient_high.unchecked_mul(divisor_low) };
    let mut middle = (trial_remainder << HALF_BITS) | numerator_mid;
    // The difference middle-high_product is the trial residue X-q*D.
    // A decrement of q adds D while high_product remains fixed. Carry from
    // that addition certifies a nonnegative difference before modular subtraction.
    if middle < high_product {
        // SAFETY: a negative trial residue puts q above a nonnegative digit.
        quotient_high = unsafe { quotient_high.unchecked_sub(1) };
        let (adjusted, carry) = middle.overflowing_add(divisor_norm);
        middle = adjusted;
        if !carry && middle < high_product {
            // SAFETY: the remaining negative trial residue still implies q>0.
            quotient_high = unsafe { quotient_high.unchecked_sub(1) };
            middle = middle.wrapping_add(divisor_norm);
        }
    }

    // Corrections put X-q*D in [0,D). Subtraction modulo one limb cancels
    // any carried limb and recovers this exact normalized residue.
    let numerator_21 = middle.wrapping_sub(high_product);

    // SAFETY: the same normalized divisor_high remains nonzero.
    let mut quotient_low = unsafe {
        numerator_21
            .checked_div(divisor_high.get())
            .unwrap_unchecked()
    };
    // SAFETY: q=floor(numerator_21/d1) gives q*d1<=numerator_21.
    let low_trial_remainder =
        unsafe { numerator_21.unchecked_sub(quotient_low.unchecked_mul(divisor_high.get())) };
    // SAFETY: numerator_21<D again gives q<=H+1 and q*d0<=H^2-1.
    let low_product = unsafe { quotient_low.unchecked_mul(divisor_low) };
    let mut lower = (low_trial_remainder << HALF_BITS) | numerator_bottom;
    if lower < low_product {
        // SAFETY: a negative trial residue requires a positive trial quotient.
        quotient_low = unsafe { quotient_low.unchecked_sub(1) };
        let (adjusted, carry) = lower.overflowing_add(divisor_norm);
        lower = adjusted;
        if !carry && lower < low_product {
            // SAFETY: the still-negative trial residue again proves q>0.
            quotient_low = unsafe { quotient_low.unchecked_sub(1) };
            lower = lower.wrapping_add(divisor_norm);
        }
    }

    // Both quotient digits are now below HALF_BASE. The normalized remainder is
    // non-negative and below divisor_norm, so shifting it back is exact.
    let quotient = (quotient_high << HALF_BITS) | quotient_low;
    let remainder_norm = lower.wrapping_sub(low_product);
    (quotient, remainder_norm >> normalization)
}
