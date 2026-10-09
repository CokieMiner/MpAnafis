//! Adjacent-tier crossover measurements.

use core::hint::black_box;
use std::time::Instant;

use mp_anafis::tune_api::{
    FormattingAlgorithm, LowProductAlgorithm, ModularPowAlgorithm, MultiplicationAlgorithm,
    MultiplicationRunner, SquaringAlgorithm,
};

use super::{
    CrossoverMeasure, FormattingPairSpec, HASH_A, HASH_B, ScoreCell, TierPairSpec, TuneSession,
    TuningProfile,
};

/// One adjacent-tier search with its ladder and measurement budget.
#[derive(Clone, Copy, Debug)]
pub struct Request<'sizes, A> {
    /// Algorithm currently occupying the interval.
    pub baseline: A,
    /// Algorithm that would replace it above the crossover.
    pub candidate: A,
    /// First width the search may accept.
    pub start: usize,
    /// Measured ladder for this transition.
    pub sizes: &'sizes [usize],
    /// Report label for this probe.
    pub tag: &'sizes str,
    /// Inner repetitions requested of the worker.
    pub iterations: u32,
}

/// Host noise and a coarse performance-state identity from one stable cell.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Coefficient of variation in parts per million.
    pub noise_cv_ppm: u32,
    /// Mean batch time rounded to milliseconds for safe cache reuse.
    pub timing_bucket_ms: u128,
}

/// Namespace grouping adjacent-tier crossover measurements and noise calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crossovers;

impl Crossovers {
    /// Estimate this host's timing noise as a coefficient of variation.
    ///
    /// Returns `stddev / mean` for repeated forced Toom-8.5 products.
    /// This calibration sets an improvement margin outside comparison timing.
    #[must_use]
    pub fn calibrate_noise() -> Calibration {
        const CALIBRATION_LEN: usize = 65_536;
        const CALIBRATION_ITERATIONS: u32 = 4;
        const CALIBRATION_SAMPLES: usize = 21;

        let left = ScoreCell::operand(CALIBRATION_LEN, HASH_A);
        let right = ScoreCell::operand(CALIBRATION_LEN, HASH_B);
        let mut destination = vec![
            0;
            CALIBRATION_LEN
                .checked_mul(2)
                .expect("calibration span fits")
        ];
        let mut runner = MultiplicationRunner::new(
            MultiplicationAlgorithm::ToomCook85,
            CALIBRATION_LEN,
            CALIBRATION_LEN,
        );
        runner.run(&mut destination, &left, &right);
        let mut prepared = runner.prepare(&mut destination, &left, &right);

        // Warm caches and page mappings before measuring batch dispersion.
        for _ in 0..CALIBRATION_SAMPLES {
            for _ in 0..CALIBRATION_ITERATIONS {
                black_box(&mut prepared).run();
            }
        }

        let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
        for _ in 0..CALIBRATION_SAMPLES {
            let started = Instant::now();
            for _ in 0..CALIBRATION_ITERATIONS {
                black_box(&mut prepared).run();
            }
            samples.push(started.elapsed().as_nanos());
        }
        let sample_count = u128::try_from(samples.len()).unwrap_or(1);
        let mean = samples.iter().sum::<u128>().div_euclid(sample_count);
        let variance = samples
            .iter()
            .map(|sample| sample.abs_diff(mean).saturating_pow(2))
            .sum::<u128>()
            .div_euclid(sample_count);
        let stddev = integer_sqrt(variance);
        let noise_cv_ppm = u32::try_from(
            stddev
                .saturating_mul(1_000_000)
                .checked_div(mean.max(1))
                .unwrap_or_else(|| u128::from(u32::MAX)),
        )
        .unwrap_or(u32::MAX);
        if noise_cv_ppm > 50_000 {
            println!(
                "WARNING: High host timing noise detected (CV {}.{:02}%). Measurements may have higher uncertainty.",
                noise_cv_ppm.div_euclid(10_000),
                noise_cv_ppm.rem_euclid(10_000).div_euclid(100)
            );
        }
        let margin_ppm = CrossoverMeasure::acceptance_margin(noise_cv_ppm);
        println!(
            "Host timing noise: CV {}.{:02}% -> acceptance margin {}.{:02}%",
            noise_cv_ppm.div_euclid(10_000),
            noise_cv_ppm.rem_euclid(10_000).div_euclid(100),
            margin_ppm.div_euclid(10_000),
            margin_ppm.rem_euclid(10_000).div_euclid(100)
        );
        Calibration {
            noise_cv_ppm,
            timing_bucket_ms: mean.div_euclid(1_000_000),
        }
    }

    /// Tune one balanced multiplication transition.
    ///
    /// Workers are compiled with `scoring_profile`, which determines the
    /// recursive child thresholds seen inside each forced tier. Callers freeze
    /// this profile for an entire tower pass so that every comparison in the
    /// same pass uses identical recursive dispatch.
    pub fn multiplication(
        session: &mut TuneSession,
        scoring_profile: &TuningProfile,
        request: Request<'_, MultiplicationAlgorithm>,
    ) -> Result<Option<usize>, String> {
        let baseline_name = multiplication_name(request.baseline)
            .ok_or_else(|| format!("Unknown baseline algorithm {:?}", request.baseline))?;
        let candidate_name = multiplication_name(request.candidate)
            .ok_or_else(|| format!("Unknown candidate algorithm {:?}", request.candidate))?;
        let mut worker_error = None;
        let crossover = CrossoverMeasure::sustained_crossover(
            request.start,
            request.sizes,
            request.tag,
            |len, quality| {
                let Some((_, _, upper_ratio)) = session.harness.score_tier_pair(
                    scoring_profile,
                    TierPairSpec {
                        family: "mul",
                        baseline: baseline_name,
                        candidate: candidate_name,
                        len,
                        quality,
                        iterations: request.iterations,
                    },
                ) else {
                    worker_error = Some(format!(
                        "worker failed measuring mul pair ({baseline_name} vs {candidate_name}) at len {len}"
                    ));
                    return false;
                };
                CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                )
            },
        );
        worker_error.map_or(Ok(crossover), Err)
    }

    /// Tune one balanced squaring transition.
    ///
    /// See [`Self::multiplication`] for the `scoring_profile` contract.
    pub fn squaring(
        session: &mut TuneSession,
        scoring_profile: &TuningProfile,
        request: Request<'_, SquaringAlgorithm>,
    ) -> Result<Option<usize>, String> {
        let baseline_name = squaring_name(request.baseline)
            .ok_or_else(|| format!("Unknown baseline algorithm {:?}", request.baseline))?;
        let candidate_name = squaring_name(request.candidate)
            .ok_or_else(|| format!("Unknown candidate algorithm {:?}", request.candidate))?;
        let mut worker_error = None;
        let crossover = CrossoverMeasure::sustained_crossover(
            request.start,
            request.sizes,
            request.tag,
            |len, quality| {
                let Some((_, _, upper_ratio)) = session.harness.score_tier_pair(
                    scoring_profile,
                    TierPairSpec {
                        family: "sqr",
                        baseline: baseline_name,
                        candidate: candidate_name,
                        len,
                        quality,
                        iterations: request.iterations,
                    },
                ) else {
                    worker_error = Some(format!(
                        "worker failed measuring sqr pair ({baseline_name} vs {candidate_name}) at len {len}"
                    ));
                    return false;
                };
                CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                )
            },
        );
        worker_error.map_or(Ok(crossover), Err)
    }

    /// Measures the schoolbook-to-Mulders or Mulders-to-full low-product crossover.
    ///
    /// See [`Self::multiplication`] for the `scoring_profile` contract: the
    /// multiplication tower inside the forced full product is frozen.
    pub fn low_product(
        session: &mut TuneSession,
        scoring_profile: &TuningProfile,
        request: Request<'_, LowProductAlgorithm>,
    ) -> Result<Option<usize>, String> {
        let algorithm_name = |algorithm| match algorithm {
            LowProductAlgorithm::Schoolbook => Some("schoolbook"),
            LowProductAlgorithm::Mulders => Some("mulders"),
            LowProductAlgorithm::Full => Some("full"),
            _ => None,
        };
        let baseline_name = algorithm_name(request.baseline)
            .ok_or_else(|| format!("Unknown baseline algorithm {:?}", request.baseline))?;
        let candidate_name = algorithm_name(request.candidate)
            .ok_or_else(|| format!("Unknown candidate algorithm {:?}", request.candidate))?;
        let mut worker_error = None;
        let crossover = CrossoverMeasure::sustained_crossover(
            request.start,
            request.sizes,
            request.tag,
            |len, quality| {
                let Some((_, _, upper_ratio)) = session.harness.score_tier_pair(
                    scoring_profile,
                    TierPairSpec {
                        family: "low",
                        baseline: baseline_name,
                        candidate: candidate_name,
                        len,
                        quality,
                        iterations: request.iterations,
                    },
                ) else {
                    worker_error = Some(format!(
                        "worker failed measuring low pair ({baseline_name} vs {candidate_name}) at len {len}"
                    ));
                    return false;
                };
                CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                )
            },
        );
        worker_error.map_or(Ok(crossover), Err)
    }

    /// Tune the schoolbook-to-recursive formatting transition.
    pub fn formatting(
        session: &mut TuneSession,
        radix: u32,
        request: Request<'_, FormattingAlgorithm>,
    ) -> Result<Option<usize>, String> {
        let algorithm_name = |algorithm| match algorithm {
            FormattingAlgorithm::Schoolbook => Some("schoolbook"),
            FormattingAlgorithm::Recursive => Some("recursive"),
            _ => None,
        };
        let baseline_name = algorithm_name(request.baseline).ok_or("Unknown baseline")?;
        let candidate_name = algorithm_name(request.candidate).ok_or("Unknown candidate")?;
        let mut worker_failed = false;
        let crossover = CrossoverMeasure::sustained_crossover(
            request.start,
            request.sizes,
            request.tag,
            |len, quality| {
                let Some((_, _, upper_ratio)) = session.harness.score_formatting_pair(
                    &session.profile,
                    FormattingPairSpec {
                        baseline: baseline_name,
                        candidate: candidate_name,
                        radix,
                        len,
                        quality,
                        iterations: request.iterations,
                    },
                ) else {
                    worker_failed = true;
                    return false;
                };
                CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                )
            },
        );
        if worker_failed {
            Err(format!("formatting worker failed for radix {radix}"))
        } else {
            Ok(crossover)
        }
    }

    /// Tune the Montgomery-to-Barrett modular exponentiation transition.
    pub fn modular_pow(
        session: &mut TuneSession,
        request: Request<'_, ModularPowAlgorithm>,
    ) -> Result<Option<usize>, String> {
        let algorithm_name = |algorithm| match algorithm {
            ModularPowAlgorithm::Montgomery => Some("montgomery"),
            ModularPowAlgorithm::Barrett => Some("barrett"),
            ModularPowAlgorithm::Production | _ => None,
        };
        let baseline_name = algorithm_name(request.baseline)
            .ok_or_else(|| format!("Unknown baseline algorithm {:?}", request.baseline))?;
        let candidate_name = algorithm_name(request.candidate)
            .ok_or_else(|| format!("Unknown candidate algorithm {:?}", request.candidate))?;
        let mut worker_error = None;
        let crossover = CrossoverMeasure::sustained_crossover(
            request.start,
            request.sizes,
            request.tag,
            |len, quality| {
                let Some((_, _, upper_ratio)) = session.harness.score_tier_pair(
                    &session.profile,
                    TierPairSpec {
                        family: "pow",
                        baseline: baseline_name,
                        candidate: candidate_name,
                        len,
                        quality,
                        iterations: request.iterations,
                    },
                ) else {
                    worker_error = Some(format!(
                        "worker failed measuring pow pair ({baseline_name} vs {candidate_name}) at len {len}"
                    ));
                    return false;
                };
                CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                )
            },
        );
        worker_error.map_or(Ok(crossover), Err)
    }
}

const fn multiplication_name(algorithm: MultiplicationAlgorithm) -> Option<&'static str> {
    match algorithm {
        MultiplicationAlgorithm::Schoolbook => Some("schoolbook"),
        MultiplicationAlgorithm::Karatsuba => Some("karatsuba"),
        MultiplicationAlgorithm::ToomCook3 => Some("toom3"),
        MultiplicationAlgorithm::ToomCook4 => Some("toom4"),
        MultiplicationAlgorithm::ToomCook6 => Some("toom6"),
        MultiplicationAlgorithm::ToomCook85 => Some("toom85"),
        #[cfg(not(target_pointer_width = "16"))]
        MultiplicationAlgorithm::SsaForced | MultiplicationAlgorithm::SsaProduction => Some("ssa"),
        _ => None,
    }
}

const fn squaring_name(algorithm: SquaringAlgorithm) -> Option<&'static str> {
    match algorithm {
        SquaringAlgorithm::Schoolbook => Some("schoolbook"),
        SquaringAlgorithm::Karatsuba => Some("karatsuba"),
        SquaringAlgorithm::ToomCook3 => Some("toom3"),
        SquaringAlgorithm::ToomCook4 => Some("toom4"),
        SquaringAlgorithm::ToomCook6 => Some("toom6"),
        SquaringAlgorithm::ToomCook85 => Some("toom85"),
        #[cfg(not(target_pointer_width = "16"))]
        SquaringAlgorithm::SsaForced | SquaringAlgorithm::SsaProduction => Some("ssa"),
        _ => None,
    }
}

/// Integer square root of a non-negative sample moment.
fn integer_sqrt(value: u128) -> u128 {
    if value <= 1 {
        return value;
    }
    let mut previous = value;
    let mut current = value.div_euclid(2).saturating_add(1);
    while current < previous {
        previous = current;
        current = previous
            .saturating_add(value.div_euclid(previous.max(1)))
            .div_euclid(2);
    }
    previous
}
