//! Adjacent-tier crossover measurements.

use core::hint::black_box;

use mp_anafis::tune_api::{Limb, formatting, multiplication, squaring};

use crate::{
    harness::{CandidateHarness, TierPairSpec},
    measure::{acceptance_margin, confidently_faster_nanos, sustained_crossover},
    tuning_profile::TuningProfile,
};

/// One adjacent-tier search with its ladder and measurement budget.
#[derive(Clone, Copy)]
pub struct Request<'sizes, A> {
    pub baseline: A,
    pub candidate: A,
    pub start: usize,
    pub sizes: &'sizes [usize],
    pub tag: &'sizes str,
    pub iterations: u32,
}

#[cfg(target_pointer_width = "64")]
const HASH_A: usize = 0x9E37_79B9_7F4A_7C15_usize;
#[cfg(target_pointer_width = "64")]
const HASH_B: usize = 0xC2B2_AE3D_27D4_EB4F_usize;
#[cfg(target_pointer_width = "32")]
const HASH_A: usize = 0x9E37_79B9_usize;
#[cfg(target_pointer_width = "32")]
const HASH_B: usize = 0xC2B2_AE3D_usize;
#[cfg(target_pointer_width = "16")]
const HASH_A: usize = 0x9E37_usize;
#[cfg(target_pointer_width = "16")]
const HASH_B: usize = 0xC2B2_usize;

/// Host noise and a coarse performance-state identity from one stable cell.
pub struct Calibration {
    /// Coefficient of variation used to derive the acceptance margin.
    pub noise_cv: f64,
    /// Mean batch time rounded to milliseconds for safe cache reuse.
    pub timing_bucket_ms: u128,
}

/// Estimate this host's timing noise as a coefficient of variation.
///
/// Times a mid-sized forced Toom-8.5 product repeatedly and returns
/// `stddev / mean`. Every crossover and compiled-knob acceptance margin is
/// derived from it, so the tuner refuses to accept differences its own machine
/// cannot distinguish. Runs outside the timed region of any comparison.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "the timing moments are computed in u128 and cast to f64 exactly for the final sqrt and ratio"
)]
pub fn calibrate_noise() -> Calibration {
    const CALIBRATION_LEN: usize = 65_536;
    const CALIBRATION_ITERATIONS: u32 = 4;
    const CALIBRATION_SAMPLES: usize = 21;

    let left = operand(CALIBRATION_LEN, HASH_A);
    let right = operand(CALIBRATION_LEN, HASH_B);
    let mut destination = vec![0; CALIBRATION_LEN.wrapping_mul(2)];
    let mut runner = multiplication::Tuner::new(
        multiplication::Algorithm::ToomCook85,
        CALIBRATION_LEN,
        CALIBRATION_LEN,
    );
    runner.run(&mut destination, &left, &right);

    let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
    for _ in 0..CALIBRATION_SAMPLES {
        let started = std::time::Instant::now();
        for _ in 0..CALIBRATION_ITERATIONS {
            runner.run(
                black_box(&mut destination),
                black_box(&left),
                black_box(&right),
            );
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
    let cv = (variance as f64).sqrt() / (mean as f64);
    println!(
        "Host timing noise: CV {:.2}% -> acceptance margin {:.2}%",
        cv * 100.0,
        f64::from(acceptance_margin(cv)) / 10_000.0
    );
    Calibration {
        noise_cv: cv,
        timing_bucket_ms: mean.div_euclid(1_000_000),
    }
}

/// Tune one balanced multiplication transition.
pub fn multiplication(
    profile: &TuningProfile,
    harness: &mut CandidateHarness,
    request: Request<'_, multiplication::Algorithm>,
    margin_ppm: u32,
) -> Option<usize> {
    let baseline_name = multiplication_name(request.baseline)?;
    let candidate_name = multiplication_name(request.candidate)?;
    sustained_crossover(request.start, request.sizes, request.tag, |len, quality| {
        let Some((baseline_time, candidate_time)) = harness.score_tier_pair(
            profile,
            TierPairSpec {
                family: "mul",
                baseline: baseline_name,
                candidate: candidate_name,
                len,
                quality,
                iterations: request.iterations,
            },
        ) else {
            return false;
        };
        confidently_faster_nanos(candidate_time, baseline_time, margin_ppm)
    })
}

/// Tune one balanced squaring transition.
pub fn squaring(
    profile: &TuningProfile,
    harness: &mut CandidateHarness,
    request: Request<'_, squaring::Algorithm>,
    margin_ppm: u32,
) -> Option<usize> {
    let baseline_name = squaring_name(request.baseline)?;
    let candidate_name = squaring_name(request.candidate)?;
    sustained_crossover(request.start, request.sizes, request.tag, |len, quality| {
        let Some((baseline_time, candidate_time)) = harness.score_tier_pair(
            profile,
            TierPairSpec {
                family: "sqr",
                baseline: baseline_name,
                candidate: candidate_name,
                len,
                quality,
                iterations: request.iterations,
            },
        ) else {
            return false;
        };
        confidently_faster_nanos(candidate_time, baseline_time, margin_ppm)
    })
}

const fn multiplication_name(algorithm: multiplication::Algorithm) -> Option<&'static str> {
    match algorithm {
        multiplication::Algorithm::Schoolbook => Some("schoolbook"),
        multiplication::Algorithm::Karatsuba => Some("karatsuba"),
        multiplication::Algorithm::ToomCook3 => Some("toom3"),
        multiplication::Algorithm::ToomCook4 => Some("toom4"),
        multiplication::Algorithm::ToomCook6 => Some("toom6"),
        multiplication::Algorithm::ToomCook85 => Some("toom85"),
        #[cfg(not(target_pointer_width = "16"))]
        multiplication::Algorithm::Ssa => Some("ssa"),
        _ => None,
    }
}

const fn squaring_name(algorithm: squaring::Algorithm) -> Option<&'static str> {
    match algorithm {
        squaring::Algorithm::Schoolbook => Some("schoolbook"),
        squaring::Algorithm::Karatsuba => Some("karatsuba"),
        squaring::Algorithm::ToomCook3 => Some("toom3"),
        squaring::Algorithm::ToomCook4 => Some("toom4"),
        squaring::Algorithm::ToomCook6 => Some("toom6"),
        squaring::Algorithm::ToomCook85 => Some("toom85"),
        #[cfg(not(target_pointer_width = "16"))]
        squaring::Algorithm::Ssa => Some("ssa"),
        _ => None,
    }
}

fn operand(len: usize, hash: usize) -> Vec<Limb> {
    (0..len).map(|index| index.wrapping_mul(hash) | 1).collect()
}

/// Tune the schoolbook-to-recursive formatting transition.
pub fn formatting(
    profile: &TuningProfile,
    harness: &mut CandidateHarness,
    radix: u32,
    request: Request<'_, formatting::Algorithm>,
    margin_ppm: u32,
) -> Result<Option<usize>, String> {
    let baseline_name = formatting_name(request.baseline).ok_or("Unknown baseline")?;
    let candidate_name = formatting_name(request.candidate).ok_or("Unknown candidate")?;
    let mut worker_failed = false;
    let crossover =
        sustained_crossover(request.start, request.sizes, request.tag, |len, quality| {
            let Some((baseline_time, candidate_time)) = harness.score_formatting_pair(
                profile,
                crate::harness::FormattingPairSpec {
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
            confidently_faster_nanos(candidate_time, baseline_time, margin_ppm)
        });
    if worker_failed {
        Err(format!("formatting worker failed for radix {radix}"))
    } else {
        Ok(crossover)
    }
}

const fn formatting_name(algorithm: formatting::Algorithm) -> Option<&'static str> {
    match algorithm {
        formatting::Algorithm::Schoolbook => Some("schoolbook"),
        formatting::Algorithm::Recursive => Some("recursive"),
        _ => None,
    }
}
