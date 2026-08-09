//! Shared normalized two-half-limb division for targets with native limb division.

use super::Limb;

/// Divide a two-limb numerator using two native limb-by-half-limb divisions.
///
/// # Safety
///
/// `divisor` must be non-zero and `rem_hi < divisor`.
#[allow(
    clippy::inline_always,
    reason = "Inlining exposes the target's native divide instruction inside the hot one-limb division loop"
)]
#[inline(always)]
pub const unsafe fn divrem_1_unchecked(limb: Limb, rem_hi: Limb, divisor: Limb) -> (Limb, Limb) {
    const HALF_BITS: u32 = Limb::BITS.wrapping_div(2);
    const HALF_BASE: Limb = 1_usize.wrapping_shl(HALF_BITS);
    const HALF_MASK: Limb = HALF_BASE.wrapping_sub(1);

    let normalization = divisor.leading_zeros();
    let divisor_norm = divisor.wrapping_shl(normalization);
    let divisor_high = divisor_norm.wrapping_shr(HALF_BITS);
    let divisor_low = divisor_norm & HALF_MASK;
    let numerator_high = if normalization == 0 {
        rem_hi
    } else {
        rem_hi.wrapping_shl(normalization)
            | limb.wrapping_shr(Limb::BITS.wrapping_sub(normalization))
    };
    let numerator_low = limb.wrapping_shl(normalization);
    let numerator_mid = numerator_low.wrapping_shr(HALF_BITS);
    let numerator_bottom = numerator_low & HALF_MASK;

    // Normalization makes divisor_high >= B/2. Since numerator_high is below
    // divisor_norm, each trial quotient is at most B+1. Knuth's correction
    // theorem therefore requires no more than two decrements.
    // Normalization of nonzero divisor makes divisor_high >= HALF_BASE / 2,
    // so checked_div is Some. Removing the impossible branch is sound.
    // SAFETY: divisor_high is mathematically proven nonzero above.
    let mut quotient_high = unsafe { numerator_high.checked_div(divisor_high).unwrap_unchecked() };
    let mut trial_remainder = numerator_high.wrapping_sub(quotient_high.wrapping_mul(divisor_high));
    let correction_high = quotient_high >= HALF_BASE
        || quotient_high.wrapping_mul(divisor_low)
            > trial_remainder
                .wrapping_shl(HALF_BITS)
                .wrapping_add(numerator_mid);
    if correction_high {
        quotient_high = quotient_high.wrapping_sub(1);
        trial_remainder = trial_remainder.wrapping_add(divisor_high);
        if trial_remainder < HALF_BASE {
            let second_correction_high = quotient_high >= HALF_BASE
                || quotient_high.wrapping_mul(divisor_low)
                    > trial_remainder
                        .wrapping_shl(HALF_BITS)
                        .wrapping_add(numerator_mid);
            if second_correction_high {
                quotient_high = quotient_high.wrapping_sub(1);
            }
        }
    }

    // The exact value is below divisor_norm after the corrected high digit;
    // wrapping arithmetic merely evaluates the cancellation modulo one limb.
    let numerator_21 = numerator_high
        .wrapping_shl(HALF_BITS)
        .wrapping_add(numerator_mid)
        .wrapping_sub(quotient_high.wrapping_mul(divisor_norm));

    // SAFETY: the same normalized divisor_high remains nonzero.
    let mut quotient_low = unsafe { numerator_21.checked_div(divisor_high).unwrap_unchecked() };
    let mut low_trial_remainder =
        numerator_21.wrapping_sub(quotient_low.wrapping_mul(divisor_high));
    let correction_low = quotient_low >= HALF_BASE
        || quotient_low.wrapping_mul(divisor_low)
            > low_trial_remainder
                .wrapping_shl(HALF_BITS)
                .wrapping_add(numerator_bottom);
    if correction_low {
        quotient_low = quotient_low.wrapping_sub(1);
        low_trial_remainder = low_trial_remainder.wrapping_add(divisor_high);
        if low_trial_remainder < HALF_BASE {
            let second_correction_low = quotient_low >= HALF_BASE
                || quotient_low.wrapping_mul(divisor_low)
                    > low_trial_remainder
                        .wrapping_shl(HALF_BITS)
                        .wrapping_add(numerator_bottom);
            if second_correction_low {
                quotient_low = quotient_low.wrapping_sub(1);
            }
        }
    }

    // Both quotient digits are now below B. The normalized remainder is
    // non-negative and below divisor_norm, so shifting it back is exact.
    let quotient = quotient_high.wrapping_shl(HALF_BITS) | quotient_low;
    let remainder_norm = numerator_21
        .wrapping_shl(HALF_BITS)
        .wrapping_add(numerator_bottom)
        .wrapping_sub(quotient_low.wrapping_mul(divisor_norm));
    (quotient, remainder_norm.wrapping_shr(normalization))
}
