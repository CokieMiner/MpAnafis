//! Adjacent-tier worker protocol parsing, correctness checks, and timing.

use core::{hint::black_box, time::Duration};

use mp_anafis::tune_api::{
    FormattingAlgorithm, FormattingRunner, GcdAlgorithm, GcdRunner, LehmerSimAlgorithm,
    LehmerSimRunner, LehmerUpdateAlgorithm, LehmerUpdateRunner, LowProductAlgorithm,
    LowProductRunner, ModularPowAlgorithm, ModularPowRunner, MultiplicationAlgorithm,
    MultiplicationRunner, SquaringAlgorithm, SquaringRunner,
};

use super::{
    CrossoverMeasure, HASH_A, HASH_B, InterleavedMeasure, PairDomain, PairSpecification, ScoreCell,
};

/// Adjacent-tier worker protocol and execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairWorkers;

impl PairWorkers {
    /// Compare two forced multiplication roots with recursively consistent thresholds.
    pub fn print_mul_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
        let baseline_algorithm = ArithmeticTier::parse(pair.baseline)
            .map(ArithmeticTier::multiplication)
            .ok_or_else(|| format!("unknown multiplication tier {}", pair.baseline))?;
        let candidate_algorithm = ArithmeticTier::parse(pair.candidate)
            .map(ArithmeticTier::multiplication)
            .ok_or_else(|| format!("unknown multiplication tier {}", pair.candidate))?;
        let left = ScoreCell::operand(pair.len, HASH_A);
        let right = ScoreCell::operand(pair.len, HASH_B);
        let output_len = pair.len.checked_mul(2).expect("validated pair width");
        let mut baseline_dst = vec![0; output_len];
        let mut candidate_dst = vec![0; output_len];
        let mut baseline_runner = MultiplicationRunner::new(baseline_algorithm, pair.len, pair.len);
        let mut candidate_runner =
            MultiplicationRunner::new(candidate_algorithm, pair.len, pair.len);
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
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));
        let (baseline_time, candidate_time, upper_ratio) = InterleavedMeasure::paired_batches(
            || black_box(&mut baseline_prepared).run(),
            || black_box(&mut candidate_prepared).run(),
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        );
        println!(
            "MP_ANAFIS_TIER_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    /// Compare two forced low-product algorithms on identical `len`-limb operands.
    pub fn print_low_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
        let parse_algorithm = |name: &str| match name {
            "schoolbook" => Ok(LowProductAlgorithm::Schoolbook),
            "mulders" => Ok(LowProductAlgorithm::Mulders),
            "full" => Ok(LowProductAlgorithm::Full),
            _ => Err(format!("unknown low product tier {name}")),
        };
        let baseline_algorithm = parse_algorithm(pair.baseline)?;
        let candidate_algorithm = parse_algorithm(pair.candidate)?;
        let left = ScoreCell::operand(pair.len, HASH_A);
        let right = ScoreCell::operand(pair.len, HASH_B);
        let mut baseline_dst = vec![0; pair.len];
        let mut candidate_dst = vec![0; pair.len];
        let mut baseline_runner = LowProductRunner::new(baseline_algorithm, pair.len);
        let mut candidate_runner = LowProductRunner::new(candidate_algorithm, pair.len);
        baseline_runner.run(&mut baseline_dst, &left, &right);
        candidate_runner.run(&mut candidate_dst, &left, &right);
        if baseline_dst != candidate_dst {
            return Err(format!(
                "low product candidates disagree at {} limbs",
                pair.len
            ));
        }
        let mut baseline_prepared = baseline_runner.prepare(&mut baseline_dst, &left, &right);
        let mut candidate_prepared = candidate_runner.prepare(&mut candidate_dst, &left, &right);
        let batch = pair
            .quality
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));
        let (baseline_time, candidate_time, upper_ratio) = InterleavedMeasure::paired_batches(
            || black_box(&mut baseline_prepared).run(),
            || black_box(&mut candidate_prepared).run(),
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        );
        println!(
            "MP_ANAFIS_TIER_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    /// Compare two forced square roots with recursively consistent thresholds.
    pub fn print_sqr_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
        let baseline_algorithm = ArithmeticTier::parse(pair.baseline)
            .and_then(ArithmeticTier::squaring)
            .ok_or_else(|| format!("unknown squaring tier {}", pair.baseline))?;
        let candidate_algorithm = ArithmeticTier::parse(pair.candidate)
            .and_then(ArithmeticTier::squaring)
            .ok_or_else(|| format!("unknown squaring tier {}", pair.candidate))?;
        let value = ScoreCell::operand(pair.len, HASH_A);
        let output_len = pair.len.checked_mul(2).expect("validated pair width");
        let mut baseline_dst = vec![0; output_len];
        let mut candidate_dst = vec![0; output_len];
        let mut baseline_runner = SquaringRunner::new(baseline_algorithm, pair.len);
        let mut candidate_runner = SquaringRunner::new(candidate_algorithm, pair.len);
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
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));
        let (baseline_time, candidate_time, upper_ratio) = InterleavedMeasure::paired_batches(
            || black_box(&mut baseline_prepared).run(),
            || black_box(&mut candidate_prepared).run(),
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        );
        println!(
            "MP_ANAFIS_TIER_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    /// Compare forced Montgomery and Barrett modular exponentiation tiers.
    pub fn print_pow_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
        let baseline_algorithm = match pair.baseline {
            "montgomery" => ModularPowAlgorithm::Montgomery,
            "barrett" => ModularPowAlgorithm::Barrett,
            _ => return Err(format!("unknown modular pow tier {}", pair.baseline)),
        };
        let candidate_algorithm = match pair.candidate {
            "montgomery" => ModularPowAlgorithm::Montgomery,
            "barrett" => ModularPowAlgorithm::Barrett,
            _ => return Err(format!("unknown modular pow tier {}", pair.candidate)),
        };
        let modulus = ScoreCell::operand(pair.len, HASH_A);
        let mut base = ScoreCell::operand(pair.len, HASH_B);
        if base >= modulus {
            let last = base.last_mut().expect("base has at least one limb");
            *last &= !(1 << (usize::BITS.saturating_sub(1)));
        }
        #[cfg(target_pointer_width = "64")]
        let exp = [0x5555_5555_5555_5555_usize];
        #[cfg(target_pointer_width = "32")]
        let exp = [0x5555_5555_usize, 0x5555_5555_usize];
        #[cfg(target_pointer_width = "16")]
        let exp = [0x5555_usize, 0x5555_usize, 0x5555_usize, 0x5555_usize];

        let mut baseline_runner = ModularPowRunner::new(&base, &exp, &modulus);
        let mut candidate_runner = ModularPowRunner::new(&base, &exp, &modulus);
        let baseline_result = baseline_runner.run(baseline_algorithm);
        let candidate_result = candidate_runner.run(candidate_algorithm);
        if baseline_result != candidate_result {
            return Err(format!(
                "modular pow candidates disagree at {} limbs",
                pair.len
            ));
        }
        let batch = pair
            .quality
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));
        let (baseline_time, candidate_time, upper_ratio) = InterleavedMeasure::paired_batches(
            || {
                drop(black_box(baseline_runner.run(baseline_algorithm)));
            },
            || {
                drop(black_box(candidate_runner.run(candidate_algorithm)));
            },
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        );
        println!(
            "MP_ANAFIS_TIER_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    /// Compare forced schoolbook and recursive formatting tiers.
    pub fn print_fmt_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Formatting)?;
        let parse_algorithm = |name: &str| match name {
            "schoolbook" => Ok(FormattingAlgorithm::Schoolbook),
            "recursive" => Ok(FormattingAlgorithm::Recursive),
            _ => Err(format!("unknown formatting tier {name}")),
        };
        let baseline_algorithm = parse_algorithm(pair.baseline)?;
        let candidate_algorithm = parse_algorithm(pair.candidate)?;
        let mut baseline_runner = FormattingRunner::new(baseline_algorithm, pair.len, pair.radix);
        let mut candidate_runner = FormattingRunner::new(candidate_algorithm, pair.len, pair.radix);
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
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));
        let (baseline_time, candidate_time, upper_ratio) = InterleavedMeasure::paired_batches(
            || baseline_runner.run(),
            || candidate_runner.run(),
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        );
        println!(
            "MP_ANAFIS_FMT_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    /// Compare forced GCD algorithmic alternatives.
    pub fn print_gcd_pair_score(specification: &str) -> Result<(), String> {
        let pair = PairSpecification::parse(specification, PairDomain::Arithmetic)?;
        let batch = pair
            .quality
            .batch(CrossoverMeasure::balanced_batch(pair.iterations, pair.len));

        let (baseline_time, candidate_time, upper_ratio) = match (pair.baseline, pair.candidate) {
            ("lehmer", "half_gcd") => Self::score_gcd_pair(&pair, batch)?,
            ("narrow", "wide") => Self::score_lehmer_sim_pair(&pair, batch)?,
            ("fused", "separate") => Self::score_lehmer_update_pair(&pair, batch)?,
            (b, c) => return Err(format!("unknown GCD pair ({b} vs {c})")),
        };

        println!(
            "MP_ANAFIS_TIER_PAIR={},{},{upper_ratio}",
            baseline_time.as_nanos(),
            candidate_time.as_nanos()
        );
        Ok(())
    }

    fn score_gcd_pair(
        pair: &PairSpecification<'_>,
        batch: u32,
    ) -> Result<(Duration, Duration, u128), String> {
        let left = ScoreCell::operand(pair.len, HASH_A);
        let right = ScoreCell::operand(pair.len, HASH_B);
        let mut baseline_runner = GcdRunner::new(&left, &right);
        let mut candidate_runner = GcdRunner::new(&left, &right);
        let baseline_res = baseline_runner.run(GcdAlgorithm::Lehmer);
        let candidate_res = candidate_runner.run(GcdAlgorithm::HalfGcd);
        if baseline_res != candidate_res {
            return Err(format!("GCD algorithms disagree at {} limbs", pair.len));
        }
        Ok(InterleavedMeasure::paired_batches(
            || {
                drop(black_box(baseline_runner.run(GcdAlgorithm::Lehmer)));
            },
            || {
                drop(black_box(candidate_runner.run(GcdAlgorithm::HalfGcd)));
            },
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        ))
    }

    fn score_lehmer_sim_pair(
        pair: &PairSpecification<'_>,
        batch: u32,
    ) -> Result<(Duration, Duration, u128), String> {
        let left = ScoreCell::operand(pair.len, HASH_A);
        let right = ScoreCell::operand(pair.len, HASH_B);
        let mut baseline_runner = LehmerSimRunner::new(&left, &right);
        let mut candidate_runner = LehmerSimRunner::new(&left, &right);
        let baseline_res = baseline_runner.run(LehmerSimAlgorithm::Narrow);
        let candidate_res = candidate_runner.run(LehmerSimAlgorithm::Wide);
        if baseline_res != candidate_res {
            return Err(format!(
                "Lehmer simulation modes disagree at {} limbs",
                pair.len
            ));
        }
        Ok(InterleavedMeasure::paired_batches(
            || {
                drop(black_box(baseline_runner.run(LehmerSimAlgorithm::Narrow)));
            },
            || {
                drop(black_box(candidate_runner.run(LehmerSimAlgorithm::Wide)));
            },
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        ))
    }

    fn score_lehmer_update_pair(
        pair: &PairSpecification<'_>,
        batch: u32,
    ) -> Result<(Duration, Duration, u128), String> {
        let left = ScoreCell::operand(pair.len, HASH_A);
        let right = ScoreCell::operand(pair.len, HASH_B);
        let mut baseline_runner = LehmerUpdateRunner::new(&left, &right);
        let mut candidate_runner = LehmerUpdateRunner::new(&left, &right);
        let baseline_res = baseline_runner.run(LehmerUpdateAlgorithm::Fused);
        let candidate_res = candidate_runner.run(LehmerUpdateAlgorithm::Separate);
        if baseline_res != candidate_res {
            return Err(format!(
                "Lehmer update modes disagree at {} limbs",
                pair.len
            ));
        }
        Ok(InterleavedMeasure::paired_batches(
            || {
                drop(black_box(baseline_runner.run(LehmerUpdateAlgorithm::Fused)));
            },
            || {
                drop(black_box(
                    candidate_runner.run(LehmerUpdateAlgorithm::Separate),
                ));
            },
            batch,
            pair.quality.samples(),
            pair.confidence_bits,
            pair.maximum_ratio,
        ))
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
    #[cfg(not(target_pointer_width = "16"))]
    SsaCrt,
    #[cfg(not(target_pointer_width = "16"))]
    SsaDirectFermat,
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
            #[cfg(not(target_pointer_width = "16"))]
            "ssa-crt" => Some(Self::SsaCrt),
            #[cfg(not(target_pointer_width = "16"))]
            "ssa-direct-fermat" => Some(Self::SsaDirectFermat),
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
            #[cfg(not(target_pointer_width = "16"))]
            Self::SsaCrt => MultiplicationAlgorithm::SsaCrt,
            #[cfg(not(target_pointer_width = "16"))]
            Self::SsaDirectFermat => MultiplicationAlgorithm::SsaDirectFermat,
        }
    }

    const fn squaring(self) -> Option<SquaringAlgorithm> {
        Some(match self {
            Self::Schoolbook => SquaringAlgorithm::Schoolbook,
            Self::Karatsuba => SquaringAlgorithm::Karatsuba,
            Self::Toom3 => SquaringAlgorithm::ToomCook3,
            Self::Toom4 => SquaringAlgorithm::ToomCook4,
            Self::Toom6 => SquaringAlgorithm::ToomCook6,
            Self::Toom85 => SquaringAlgorithm::ToomCook85,
            #[cfg(not(target_pointer_width = "16"))]
            Self::Ssa => SquaringAlgorithm::SsaForced,
            Self::SsaProduction => SquaringAlgorithm::SsaProduction,
            #[cfg(not(target_pointer_width = "16"))]
            Self::SsaCrt | Self::SsaDirectFermat => return None,
        })
    }
}
