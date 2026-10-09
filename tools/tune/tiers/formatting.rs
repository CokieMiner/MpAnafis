//! Schoolbook-to-recursive radix formatting crossover tuning and verification.

use core::ops::RangeInclusive;

use mp_anafis::tune_api::FormattingAlgorithm;

use super::{
    CrossoverMeasure, Crossovers, FormattingPairSpec, ITERATIONS, ProbeQuality, Request,
    TuneSession,
};

const FORMAT_SIZES: [usize; 24] = [
    4, 5, 6, 7, 8, 12, 16, 24, 32, 40, 48, 49, 50, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
    768,
];

/// Tuning driver for schoolbook-to-recursive radix formatting crossovers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormattingTuner;

impl FormattingTuner {
    /// Tune the schoolbook-to-recursive formatting crossover.
    pub fn tune_formatting(session: &mut TuneSession) {
        println!("\nRadix formatting crossovers");

        let mut tune_one = |radix: u32,
                            fallback: usize,
                            inner_session: &mut TuneSession|
         -> Result<usize, String> {
            let tag = format!("Schoolbook -> Recursive formatting (radix {radix})");
            let result = Crossovers::formatting(
                inner_session,
                radix,
                Request {
                    baseline: FormattingAlgorithm::Schoolbook,
                    candidate: FormattingAlgorithm::Recursive,
                    start: 4,
                    sizes: &FORMAT_SIZES,
                    tag: &tag,
                    iterations: ITERATIONS,
                },
            )?;
            Ok(result.unwrap_or(fallback))
        };

        let result: Result<[usize; 3], String> = (|| {
            let mut thresholds = [0; 3];
            for (threshold, (radices, fallback)) in thresholds.iter_mut().zip([
                (10..=10, session.profile.radix_format_decimal_recursive),
                (3..=9, session.profile.radix_format_small_recursive),
                (11..=36, session.profile.radix_format_large_recursive),
            ]) {
                let first = tune_one(*radices.start(), fallback, session)?;
                let last = if radices.start() == radices.end() {
                    first
                } else {
                    tune_one(*radices.end(), fallback, session)?
                };
                *threshold =
                    validate_group(session, first.max(last), radices, fallback, &mut tune_one)?;
            }
            Ok(thresholds)
        })();
        let [cross_10, small_threshold, large_threshold] = match result {
            Ok(thresholds) => thresholds,
            Err(error) => {
                session.harness.reject(&error);
                session.record(
                    "RADIX_FORMATTING",
                    format!("failed: {error}; input thresholds retained"),
                );
                return;
            }
        };

        session.profile.radix_format_decimal_recursive = cross_10;
        session.profile.radix_format_small_recursive = small_threshold;
        session.profile.radix_format_large_recursive = large_threshold;

        session.record(
            "RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD".to_owned(),
            format!("{}", session.profile.radix_format_decimal_recursive),
        );
        session.record(
            "RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD".to_owned(),
            format!("{}", session.profile.radix_format_small_recursive),
        );
        session.record(
            "RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD".to_owned(),
            format!("{}", session.profile.radix_format_large_recursive),
        );
    }
}

fn validate_group(
    session: &mut TuneSession,
    mut threshold: usize,
    radices: RangeInclusive<u32>,
    fallback: usize,
    tune_one: &mut impl FnMut(u32, usize, &mut TuneSession) -> Result<usize, String>,
) -> Result<usize, String> {
    if threshold == usize::MAX - 1 {
        return Ok(threshold);
    }
    loop {
        let mut failed_radix = None;
        for radix in radices.clone() {
            if radix.is_power_of_two() {
                continue;
            }
            for len in [
                threshold,
                threshold.saturating_add(4),
                threshold.saturating_add(8),
            ] {
                let spec = FormattingPairSpec {
                    baseline: "schoolbook",
                    candidate: "recursive",
                    radix,
                    len,
                    quality: ProbeQuality::Precise,
                    iterations: ITERATIONS,
                };
                let Some((_, _, upper_ratio)) = session
                    .harness
                    .score_formatting_pair(&session.profile, spec)
                else {
                    return Err(format!("Could not validate formatting radix {radix}"));
                };
                if !CrossoverMeasure::confidently_faster_nanos(
                    upper_ratio,
                    1_000_000,
                    session.margin_ppm,
                ) {
                    failed_radix = Some(radix);
                    break;
                }
            }
            if failed_radix.is_some() {
                break;
            }
        }
        if let Some(radix) = failed_radix {
            let new_cross = tune_one(radix, fallback, session)?;
            if new_cross <= threshold {
                println!(
                    "Formatting validation for radix {radix} failed at {threshold}, \
                     but retuning produced no later crossover"
                );
                return Ok(fallback);
            }
            threshold = new_cross;
        } else {
            break;
        }
    }
    Ok(threshold)
}
