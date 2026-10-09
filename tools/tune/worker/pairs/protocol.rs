//! Adjacent-tier worker arguments and allocation-width validation.

use core::mem::size_of;

use super::ProbeQuality;

/// Argument layout for one adjacent-tier comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairDomain {
    /// Multiplication, low product, squaring, modular exponentiation, or GCD.
    Arithmetic,
    /// Radix formatting with an explicit radix field.
    Formatting,
}

/// Parsed adjacent-tier worker arguments.
pub struct PairSpecification<'specification> {
    pub baseline: &'specification str,
    pub candidate: &'specification str,
    pub radix: u32,
    pub len: usize,
    pub quality: ProbeQuality,
    pub iterations: u32,
    pub confidence_bits: u32,
    pub maximum_ratio: u128,
}

impl<'specification> PairSpecification<'specification> {
    /// Parses a complete worker request before operand or output allocation.
    pub fn parse(specification: &'specification str, domain: PairDomain) -> Result<Self, String> {
        let mut fields = specification.split(',');
        let baseline = fields.next().ok_or("missing baseline tier")?;
        let candidate = fields.next().ok_or("missing candidate tier")?;
        let radix = if matches!(domain, PairDomain::Formatting) {
            fields
                .next()
                .ok_or("missing formatting radix")?
                .parse::<u32>()
                .map_err(|error| format!("invalid formatting radix: {error}"))?
        } else {
            0
        };
        let len = fields
            .next()
            .ok_or("missing tier width")?
            .parse::<usize>()
            .map_err(|error| format!("invalid tier width: {error}"))?;
        let quality = match fields.next() {
            Some("coarse") => ProbeQuality::Coarse,
            Some("precise") => ProbeQuality::Precise,
            _ => return Err("tier quality must be coarse or precise".to_owned()),
        };
        let iterations = fields
            .next()
            .ok_or("missing tier iterations")?
            .parse::<u32>()
            .map_err(|error| format!("invalid tier iterations: {error}"))?;
        let confidence_bits = fields
            .next()
            .ok_or("missing confidence budget")?
            .parse::<u32>()
            .map_err(|error| format!("invalid confidence budget: {error}"))?;
        let maximum_ratio = fields
            .next()
            .ok_or("missing acceptance limit")?
            .parse::<u128>()
            .map_err(|error| format!("invalid acceptance limit: {error}"))?;
        let invalid_radix = matches!(domain, PairDomain::Formatting)
            && (!(3..=36).contains(&radix) || radix.is_power_of_two());
        let invalid_width = len
            .checked_mul(2)
            .and_then(|limbs| limbs.checked_mul(size_of::<usize>()))
            .is_none_or(|bytes| isize::try_from(bytes).is_err());
        if fields.next().is_some()
            || len == 0
            || iterations == 0
            || invalid_radix
            || invalid_width
            || maximum_ratio == 0
            || maximum_ratio >= 1_000_000
            || (quality == ProbeQuality::Coarse) != (confidence_bits == 0)
        {
            return Err(match domain {
                PairDomain::Arithmetic => "invalid tier-pair specification".to_owned(),
                PairDomain::Formatting => "invalid formatting tier-pair specification".to_owned(),
            });
        }
        Ok(Self {
            baseline,
            candidate,
            radix,
            len,
            quality,
            iterations,
            confidence_bits,
            maximum_ratio,
        })
    }
}
