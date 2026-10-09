//! Integer-to-float conversion with round-to-nearest, ties-to-even.

#![expect(
    unsafe_code,
    reason = "Normalized input, the 1024-bit limit, and fixed IEEE-754 precisions bound initialized limb extraction and exponent arithmetic"
)]

use super::{InternalMpUint, LIMB_BITS, Limb};

impl InternalMpUint {
    /// Converts the value to `f64`, returning `None` if the value is too large
    /// (f64 overflow).
    ///
    /// A single limb uses native rounding; wider inputs extract the leading
    /// significand and inspect discarded bits for ties-to-even rounding.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        let limbs = self.limbs();
        let n = limbs.len();
        if n == 0 {
            return Some(0.0);
        }
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "Native integer-to-float conversion supplies the required ties-to-even rounding; every supported limb is at most 64 bits and fits the f64 exponent range."
        )]
        if let [limb] = limbs {
            return Some(*limb as f64);
        }
        // Every finite f64 integer occupies at most 1024 bits. Reject by
        // limb count before a potentially unrepresentable full bit length.
        if n > 1024_usize.div_ceil(LIMB_BITS) {
            return None;
        }
        let (significand, exponent) = rounded_float_components(limbs, 53, 1023)?;
        #[expect(
            clippy::as_conversions,
            reason = "The returned exponent is at most 1023; biasing yields at most 2046, fitting usize and u64 on every pointer width"
        )]
        {
            let mantissa_bits = significand & 0x000F_FFFF_FFFF_FFFF;
            // SAFETY: exponent <= 1023, hence exponent + 1023 <= 2046
            // fits even 16-bit usize, and its lossless u64 cast has 11 bits.
            let biased_exponent = unsafe { exponent.unchecked_add(1023) };
            let bits_u64 = (biased_exponent as u64) << 52 | mantissa_bits;
            Some(f64::from_bits(bits_u64))
        }
    }

    /// Converts the value to `f32`, returning `None` if the value is too large
    /// (f32 overflow).
    ///
    /// Limbs of at most 32 bits use native rounding. Wider inputs extract the
    /// leading significand and inspect discarded bits for ties-to-even rounding.
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        let limbs = self.limbs();
        let n = limbs.len();
        if n == 0 {
            return Some(0.0);
        }
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "A 16- or 32-bit limb is exact in f64, so even a native conversion through f64 rounds only once to f32"
        )]
        if let [limb] = limbs {
            return Some(*limb as f32);
        }
        // Every finite f32 integer occupies at most 128 bits.
        if n > 128_usize.div_ceil(LIMB_BITS) {
            return None;
        }
        let (significand, exponent) = rounded_float_components(limbs, 24, 127)?;
        let mantissa_bits = significand & 0x007F_FFFF;
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The returned exponent is at most 127; biasing yields at most 254 on every pointer width, and the mantissa is masked to 23 bits"
        )]
        let bits_u32 = {
            // SAFETY: exponent <= 127, so exponent + 127 <= 254 fits usize
            // on 16-, 32-, and 64-bit targets and converts losslessly to u32.
            let biased_exponent = unsafe { exponent.unchecked_add(127) };
            (biased_exponent as u32) << 23 | (mantissa_bits as u32)
        };
        Some(f32::from_bits(bits_u32))
    }
}

/// Returns the rounded binary significand and unbiased exponent for an
/// integer-to-float conversion. Rounding follows round-to-nearest, ties-to-even.
/// Callers supply a nonempty normalized slice of at most 1024 bits and
/// precision 53 or 24, after handling zero and excessive length.
fn rounded_float_components(
    limbs: &[Limb],
    precision: usize,
    max_exponent: usize,
) -> Option<(u64, usize)> {
    debug_assert!(precision < 64, "float significands must fit in u64");
    // SAFETY: each float entry point handles zero before calling this helper,
    // so the normalized slice has an initialized last limb.
    let (&top, lower) = unsafe { limbs.split_last().unwrap_unchecked() };
    // SAFETY: normalization gives top != 0, so leading_zeros < LIMB_BITS <= 64.
    // Its conversion fits even 16-bit usize, and the difference is positive.
    let top_bits =
        unsafe { LIMB_BITS.unchecked_sub(usize::try_from(top.leading_zeros()).unwrap_unchecked()) };
    // SAFETY: the caller bounds len <= ceil(1024 / LIMB_BITS), so the exact
    // bit count lies in 1..=1024, fitting usize on every supported target.
    let bits = unsafe { lower.len().unchecked_mul(LIMB_BITS).unchecked_add(top_bits) };
    // SAFETY: top_bits > 0 proves bits > 0.
    let mut exponent = unsafe { bits.unchecked_sub(1) };
    if exponent > max_exponent {
        return None;
    }

    let mut significand = leading_bits_as_u64(limbs, bits, precision);
    if bits > precision {
        // SAFETY: bits > precision, so discarded >= 1 and round_bit >= 0.
        // Both differences are bounded by bits <= 1024 on every target.
        let round_bit = unsafe { bits.unchecked_sub(precision).unchecked_sub(1) };
        if should_round_up(limbs, round_bit, significand) {
            // SAFETY: extraction yields significand < 2^precision, with
            // precision <= 53; adding one is at most 2^53 < u64::MAX.
            significand = unsafe { significand.unchecked_add(1) };
            // Callers fix precision at 53 or 24, strictly below u64::BITS.
            if significand == 1_u64 << precision {
                significand >>= 1;
                // SAFETY: the preceding exponent guard bounds exponent by
                // max_exponent <= 1023, so adding one fits even 16-bit usize.
                exponent = unsafe { exponent.unchecked_add(1) };
                if exponent > max_exponent {
                    return None;
                }
            }
        }
    }

    Some((significand, exponent))
}

#[inline]
#[expect(
    clippy::as_conversions,
    reason = "The nonzero normalized slice bounds its leading suffix; widening 16-, 32-, or 64-bit limbs to u64 is lossless"
)]
fn leading_bits_as_u64(limbs: &[Limb], significant_bits: usize, width: usize) -> u64 {
    let discarded = significant_bits.saturating_sub(width);
    let first_index = discarded.div_euclid(LIMB_BITS);
    let offset = discarded.rem_euclid(LIMB_BITS);
    // SAFETY: width is 24 or 53 and significant_bits > 0. Hence discarded
    // is strictly below the bit length and first_index lies within limbs.
    let leading = unsafe { limbs.get_unchecked(first_index..) };
    // SAFETY: the preceding bound proves this suffix contains at least one limb.
    let (&first, rest) = unsafe { leading.split_first().unwrap_unchecked() };
    // Normalization bounds the shifted suffix to min(width, significant_bits)
    // bits. Whole high limbs can therefore be appended without individual masks.
    let mut acc = (first as u64) >> offset;
    // SAFETY: offset < LIMB_BITS by remainder, so the difference is positive.
    let mut filled = unsafe { LIMB_BITS.unchecked_sub(offset) };
    for &limb in rest {
        // Another stored limb starts below the final width <= 53, so filled
        // is below 53 whenever it is used as a shift, on every pointer width.
        acc |= (limb as u64) << filled;
        // SAFETY: a remaining stored limb starts below width <= 53; therefore
        // filled < 53 before this addition and filled + LIMB_BITS <= 116,
        // fitting usize on 16-, 32-, and 64-bit targets.
        filled = unsafe { filled.unchecked_add(LIMB_BITS) };
    }
    acc << width.saturating_sub(significant_bits)
}

/// Tests the discarded round bit and sticky prefix for ties-to-even rounding.
#[inline]
fn should_round_up(limbs: &[Limb], round_bit: usize, significand: u64) -> bool {
    let index = round_bit.div_euclid(LIMB_BITS);
    let offset = round_bit.rem_euclid(LIMB_BITS);
    // SAFETY: round_bit < significant_bits, so index < limbs.len(). The
    // immutable slice is initialized, and the remainder bounds the shift.
    let limb = unsafe { *limbs.get_unchecked(index) };
    if (limb >> offset) & 1 == 0 {
        return false;
    }
    if significand & 1 != 0 {
        return true;
    }
    // SAFETY: offset < LIMB_BITS makes 1 << offset a positive representable
    // Limb. Subtracting one gives exactly the lower-offset-bit mask, including
    // zero when the round bit is the first bit in its limb.
    let mask = unsafe { (1_usize << offset).unchecked_sub(1) };
    // SAFETY: index < limbs.len() bounds the initialized lower prefix.
    let lower = unsafe { limbs.get_unchecked(..index) };
    limb & mask != 0 || lower.iter().any(|&low| low != 0)
}
