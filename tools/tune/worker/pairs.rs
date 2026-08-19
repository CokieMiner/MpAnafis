//! Adjacent-tier worker protocol parsing, correctness checks, and timing.

use core::hint::black_box;

use mp_anafis::tune_api::{FormattingAlgorithm, MultiplicationAlgorithm, SquaringAlgorithm, Tuner};

use crate::measure::{ProbeQuality, balanced_batch, median_pair_batches};

use super::{HASH_A, HASH_B, operand};

/// Compare two forced multiplication roots with recursively consistent thresholds.
pub fn print_mul_pair_score(specification: &str) -> Result<(), String> {
    let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
    let baseline_algorithm = ArithmeticTier::parse(pair.baseline)
        .map(ArithmeticTier::multiplication)
        .ok_or_else(|| format!("unknown multiplication tier {}", pair.baseline))?;
    let candidate_algorithm = ArithmeticTier::parse(pair.candidate)
        .map(ArithmeticTier::multiplication)
        .ok_or_else(|| format!("unknown multiplication tier {}", pair.candidate))?;
    let left = operand(pair.len, HASH_A);
    let right = operand(pair.len, HASH_B);
    let mut baseline_dst = vec![0; pair.len.wrapping_mul(2)];
    let mut candidate_dst = vec![0; pair.len.wrapping_mul(2)];
    let mut baseline_runner = Tuner::multiplication(baseline_algorithm, pair.len, pair.len);
    let mut candidate_runner = Tuner::multiplication(candidate_algorithm, pair.len, pair.len);
    baseline_runner.run(&mut baseline_dst, &left, &right);
    candidate_runner.run(&mut candidate_dst, &left, &right);
    if baseline_dst != candidate_dst {
        return Err(format!(
            "multiplication candidates disagree at {} limbs",
            pair.len
        ));
    }
    let mut baseline_prepared = baseline_runner.prepare(&mut baseline_dst, &left, &right);
    let mut candidate_prepared = candidate_runner.prepare(&mut candidate_dst, &left, &right);
    let batch = pair
        .quality
        .batch(balanced_batch(pair.iterations, pair.len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || black_box(&mut baseline_prepared).run(),
        || black_box(&mut candidate_prepared).run(),
        batch,
        pair.quality.samples(),
    );
    println!(
        "MP_ANAFIS_TIER_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

/// Compare two forced square roots with recursively consistent thresholds.
pub fn print_sqr_pair_score(specification: &str) -> Result<(), String> {
    let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
    let baseline_algorithm = ArithmeticTier::parse(pair.baseline)
        .map(ArithmeticTier::squaring)
        .ok_or_else(|| format!("unknown squaring tier {}", pair.baseline))?;
    let candidate_algorithm = ArithmeticTier::parse(pair.candidate)
        .map(ArithmeticTier::squaring)
        .ok_or_else(|| format!("unknown squaring tier {}", pair.candidate))?;
    let value = operand(pair.len, HASH_A);
    let mut baseline_dst = vec![0; pair.len.wrapping_mul(2)];
    let mut candidate_dst = vec![0; pair.len.wrapping_mul(2)];
    let mut baseline_runner = Tuner::squaring(baseline_algorithm, pair.len);
    let mut candidate_runner = Tuner::squaring(candidate_algorithm, pair.len);
    baseline_runner.run(&mut baseline_dst, &value);
    candidate_runner.run(&mut candidate_dst, &value);
    if baseline_dst != candidate_dst {
        return Err(format!(
            "squaring candidates disagree at {} limbs",
            pair.len
        ));
    }
    let mut baseline_prepared = baseline_runner.prepare(&mut baseline_dst, &value);
    let mut candidate_prepared = candidate_runner.prepare(&mut candidate_dst, &value);
    let batch = pair
        .quality
        .batch(balanced_batch(pair.iterations, pair.len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || black_box(&mut baseline_prepared).run(),
        || black_box(&mut candidate_prepared).run(),
        batch,
        pair.quality.samples(),
    );
    println!(
        "MP_ANAFIS_TIER_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

/// Compare forced schoolbook and recursive formatting tiers.
pub fn print_fmt_pair_score(specification: &str) -> Result<(), String> {
    let pair = PairSpecification::parse(specification, PairDomain::Formatting)?;
    let baseline_algorithm = formatting_algorithm(pair.baseline)?;
    let candidate_algorithm = formatting_algorithm(pair.candidate)?;
    let mut baseline_runner = Tuner::formatting(baseline_algorithm, pair.len, pair.radix);
    let mut candidate_runner = Tuner::formatting(candidate_algorithm, pair.len, pair.radix);
    let baseline_output = baseline_runner.output();
    let candidate_output = candidate_runner.output();
    if baseline_output != candidate_output {
        return Err(format!(
            "formatting candidates disagree at {} limbs: schoolbook={}, recursive={}",
            pair.len,
            baseline_output.len(),
            candidate_output.len()
        ));
    }
    let batch = pair
        .quality
        .batch(balanced_batch(pair.iterations, pair.len));
    let (baseline_time, candidate_time) = median_pair_batches(
        || baseline_runner.run(),
        || candidate_runner.run(),
        batch,
        pair.quality.samples(),
    );
    println!(
        "MP_ANAFIS_FMT_PAIR={},{}",
        baseline_time.as_nanos(),
        candidate_time.as_nanos()
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum PairDomain {
    Arithmetic,
    Formatting,
}

struct PairSpecification<'specification> {
    baseline: &'specification str,
    candidate: &'specification str,
    radix: u32,
    len: usize,
    quality: ProbeQuality,
    iterations: u32,
}

impl<'specification> PairSpecification<'specification> {
    fn parse(specification: &'specification str, domain: PairDomain) -> Result<Self, String> {
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
        let invalid_radix = matches!(domain, PairDomain::Formatting)
            && (!(3..=36).contains(&radix) || radix.is_power_of_two());
        if fields.next().is_some() || len == 0 || iterations == 0 || invalid_radix {
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
        })
    }
}

#[derive(Clone, Copy)]
enum ArithmeticTier {
    Schoolbook,
    Karatsuba,
    Toom3,
    Toom4,
    Toom6,
    Toom85,
    #[cfg(not(target_pointer_width = "16"))]
    Ssa,
    SsaProduction,
}

impl ArithmeticTier {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "schoolbook" => Some(Self::Schoolbook),
            "karatsuba" => Some(Self::Karatsuba),
            "toom3" => Some(Self::Toom3),
            "toom4" => Some(Self::Toom4),
            "toom6" => Some(Self::Toom6),
            "toom85" => Some(Self::Toom85),
            #[cfg(not(target_pointer_width = "16"))]
            "ssa" => Some(Self::Ssa),
            "ssa-production" => Some(Self::SsaProduction),
            _ => None,
        }
    }

    const fn multiplication(self) -> MultiplicationAlgorithm {
        match self {
            Self::Schoolbook => MultiplicationAlgorithm::Schoolbook,
            Self::Karatsuba => MultiplicationAlgorithm::Karatsuba,
            Self::Toom3 => MultiplicationAlgorithm::ToomCook3,
            Self::Toom4 => MultiplicationAlgorithm::ToomCook4,
            Self::Toom6 => MultiplicationAlgorithm::ToomCook6,
            Self::Toom85 => MultiplicationAlgorithm::ToomCook85,
            #[cfg(not(target_pointer_width = "16"))]
            Self::Ssa => MultiplicationAlgorithm::SsaForced,
            Self::SsaProduction => MultiplicationAlgorithm::SsaProduction,
        }
    }

    const fn squaring(self) -> SquaringAlgorithm {
        match self {
            Self::Schoolbook => SquaringAlgorithm::Schoolbook,
            Self::Karatsuba => SquaringAlgorithm::Karatsuba,
            Self::Toom3 => SquaringAlgorithm::ToomCook3,
            Self::Toom4 => SquaringAlgorithm::ToomCook4,
            Self::Toom6 => SquaringAlgorithm::ToomCook6,
            Self::Toom85 => SquaringAlgorithm::ToomCook85,
            #[cfg(not(target_pointer_width = "16"))]
            Self::Ssa => SquaringAlgorithm::SsaForced,
            Self::SsaProduction => SquaringAlgorithm::SsaProduction,
        }
    }
}

fn formatting_algorithm(name: &str) -> Result<FormattingAlgorithm, String> {
    match name {
        "schoolbook" => Ok(FormattingAlgorithm::Schoolbook),
        "recursive" => Ok(FormattingAlgorithm::Recursive),
        _ => Err(format!("unknown formatting tier {name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{PairDomain, PairSpecification};
    use crate::measure::ProbeQuality;

    #[test]
    fn arithmetic_protocol_preserves_field_order() {
        let pair =
            PairSpecification::parse("schoolbook,karatsuba,128,coarse,9", PairDomain::Arithmetic)
                .expect("valid arithmetic pair");
        assert_eq!(pair.baseline, "schoolbook");
        assert_eq!(pair.candidate, "karatsuba");
        assert_eq!(pair.radix, 0);
        assert_eq!(pair.len, 128);
        assert_eq!(pair.quality, ProbeQuality::Coarse);
        assert_eq!(pair.iterations, 9);
    }

    #[test]
    fn formatting_protocol_preserves_radix_before_width() {
        let pair = PairSpecification::parse(
            "schoolbook,recursive,10,64,precise,5",
            PairDomain::Formatting,
        )
        .expect("valid formatting pair");
        assert_eq!(pair.radix, 10);
        assert_eq!(pair.len, 64);
        assert_eq!(pair.quality, ProbeQuality::Precise);
        assert_eq!(pair.iterations, 5);
    }

    #[test]
    fn formatting_protocol_rejects_power_of_two_and_extra_fields() {
        assert!(
            PairSpecification::parse(
                "schoolbook,recursive,8,64,precise,5",
                PairDomain::Formatting,
            )
            .is_err(),
            "power-of-two radix must be rejected"
        );
        assert!(
            PairSpecification::parse(
                "schoolbook,recursive,10,64,precise,5,extra",
                PairDomain::Formatting,
            )
            .is_err(),
            "extra protocol fields must be rejected"
        );
    }
}
