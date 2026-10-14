//! Geometry selection from coefficient bounds and finite-size arithmetic work.

use super::{NttProfile, STANDARD_PROFILES};

/// Resolved digit lengths, transform length, and prime count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedProfile {
    /// Selected coefficient width and modulus count.
    pub profile: NttProfile,
    /// Active coefficients in the left operand.
    pub digits_a: usize,
    /// Active coefficients in the right operand.
    pub digits_b: usize,
    /// Number of nonzero convolution coefficient slots.
    pub conv_len: usize,
    /// Power-of-two transform length.
    pub transform_len: usize,
    /// Base-two logarithm of the transform length.
    pub transform_log: u32,
}

impl PlannedProfile {
    /// Selects a plan by total transform and reconstruction work.
    ///
    /// Every nonempty catalog prefix is considered. Ties prefer fewer primes.
    /// Returns `None` for zero operands, invalid limb widths, or overflowing sizes.
    #[must_use]
    pub fn select(len_a: usize, len_b: usize, limb_bits: usize) -> Option<Self> {
        if !matches!(limb_bits, 16 | 32 | 64) || len_a == 0 || len_b == 0 {
            return None;
        }
        let bits_a = len_a.checked_mul(limb_bits)?;
        let bits_b = len_b.checked_mul(limb_bits)?;
        let mut best: Option<(u128, Self)> = None;
        for mut profile in STANDARD_PROFILES {
            while let Some(candidate) = geometry(bits_a, bits_b, profile) {
                let terms = candidate.digits_a.min(candidate.digits_b);
                if profile.admits(candidate.transform_len, terms) {
                    // Three radix-2-equivalent transforms and np*(np-1)/2
                    // mixed-radix steps per coefficient. This is a work model,
                    // not a measured claim about a particular processor.
                    let count = u128::try_from(profile.np).ok()?;
                    let length = u128::try_from(candidate.transform_len).ok()?;
                    let coefficients = u128::try_from(candidate.conv_len).ok()?;
                    let transform = count
                        .checked_mul(length)?
                        .checked_mul(u128::from(candidate.transform_log))?
                        .checked_mul(3)?;
                    let crt = count
                        .checked_mul(count.checked_sub(1)?)?
                        .checked_mul(coefficients)?;
                    let score = transform.checked_add(crt)?;
                    if best.is_none_or(|(cost, _)| score < cost) {
                        best = Some((score, candidate));
                    }
                    break;
                }
                let log_terms = usize::BITS.checked_sub(terms.saturating_sub(1).leading_zeros())?;
                let capacity = u32::try_from(profile.np).ok()?.checked_mul(49)?;
                let needed = profile.chunk_bits.checked_mul(2)?.checked_add(log_terms)?;
                let excess = needed.saturating_sub(capacity).max(1);
                let Some(narrower) = profile.chunk_bits.checked_sub(excess.div_ceil(2)) else {
                    break;
                };
                if narrower == 0 {
                    break;
                }
                profile.chunk_bits = narrower;
            }
        }
        best.map(|(_, candidate)| candidate)
    }
}

fn geometry(bits_a: usize, bits_b: usize, profile: NttProfile) -> Option<PlannedProfile> {
    let width = usize::try_from(profile.chunk_bits).ok()?;
    if width == 0 {
        return None;
    }
    let digits_a = bits_a.div_ceil(width);
    let digits_b = bits_b.div_ceil(width);
    let conv_len = digits_a.checked_add(digits_b)?.checked_sub(1)?;
    let transform_len = conv_len.checked_next_power_of_two()?.max(4);
    Some(PlannedProfile {
        profile,
        digits_a,
        digits_b,
        conv_len,
        transform_len,
        transform_log: transform_len.trailing_zeros(),
    })
}
