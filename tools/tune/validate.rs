//! End-to-end production-dispatch validation.
//!
//! Every earlier phase measures *forced* tiers, which can hide a bad crossover
//! or a profile whose parts disagree with each other. This gate scores the
//! real production multiplication, squaring, and division dispatchers with the
//! tuned profile and with the architecture defaults. The tuned profile must
//! beat the defaults by the host noise margin; otherwise the run is rejected
//! and nothing is installed.

use crate::{
    harness::{SCORE_SCALE, ScoreDomain, relative_score},
    session::TuneSession,
    worker::{
        PRODUCTION_DIV_CELLS, PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, cell_weights,
        division_cell_weights,
    },
};

/// Score the tuned profile against the architecture defaults on the
/// production dispatcher. Returns `true` when the tuned profile wins by the
/// margin, and records the per-cell verdicts in `decisions`.
pub fn end_to_end(session: &mut TuneSession) -> bool {
    println!("\nEnd-to-end validation (production dispatcher)");
    let mut weights = cell_weights(&PRODUCTION_MUL_CELLS, &PRODUCTION_SQR_CELLS);
    weights.extend(division_cell_weights());
    let Some(tuned) = session
        .harness
        .score(&session.profile, ScoreDomain::Production)
    else {
        println!("Validation failed: could not score the tuned profile");
        return false;
    };
    let Some(baseline) = session
        .harness
        .score(&session.defaults, ScoreDomain::Production)
    else {
        println!("Validation failed: could not score the architecture defaults");
        return false;
    };
    let tuned_score = relative_score(&tuned, &baseline, &weights);
    let baseline_score = SCORE_SCALE;
    let mul_weights = cell_weights(&PRODUCTION_MUL_CELLS, &[]);
    let sqr_weights = cell_weights(&[], &PRODUCTION_SQR_CELLS);
    let div_weights = division_cell_weights();
    let mul_score = segment_score(&tuned, &baseline, 0, &mul_weights);
    let sqr_offset = PRODUCTION_MUL_CELLS.len();
    let sqr_score = segment_score(&tuned, &baseline, sqr_offset, &sqr_weights);
    let div_offset = sqr_offset.saturating_add(PRODUCTION_SQR_CELLS.len());
    let div_score = segment_score(&tuned, &baseline, div_offset, &div_weights);
    report_cells(
        &tuned,
        &baseline,
        &PRODUCTION_MUL_CELLS,
        &PRODUCTION_SQR_CELLS,
        &PRODUCTION_DIV_CELLS,
    );

    let aggregate_wins = tuned_score.saturating_mul(1_000_000)
        < baseline_score
            .saturating_mul(u128::from(1_000_000_u32.saturating_sub(session.margin_ppm)));
    let family_limit = SCORE_SCALE.saturating_add(u128::from(session.margin_ppm));
    let no_family_regresses = [mul_score, sqr_score, div_score]
        .into_iter()
        .all(|score| score <= family_limit);
    println!(
        "  family scores: multiplication {mul_score} ppm, square {sqr_score} ppm, division {div_score} ppm"
    );
    let wins = aggregate_wins && no_family_regresses;
    if wins {
        println!("Validation passed: tuned profile {tuned_score} ppm vs defaults {baseline_score}");
        session.record(
            "END_TO_END_VALIDATION".to_owned(),
            format!("passed at {tuned_score} ppm"),
        );
    } else {
        let margin_ppm = session.margin_ppm;
        println!(
            "Validation REJECTED: tuned profile {tuned_score} ppm vs defaults {baseline_score} \
             (needed {margin_ppm} ppm aggregate win and no family above {family_limit} ppm); \
             not installing the tuned profile"
        );
        session.record(
            "END_TO_END_VALIDATION".to_owned(),
            format!("rejected: {tuned_score} ppm, defaults {baseline_score}"),
        );
    }
    wins
}

fn segment_score(tuned: &[u128], baseline: &[u128], start: usize, weights: &[u32]) -> u128 {
    let Some(end) = start.checked_add(weights.len()) else {
        return u128::MAX;
    };
    let Some(tuned_segment) = tuned.get(start..end) else {
        return u128::MAX;
    };
    let Some(baseline_segment) = baseline.get(start..end) else {
        return u128::MAX;
    };
    relative_score(tuned_segment, baseline_segment, weights)
}

fn report_cells(
    tuned: &[u128],
    baseline: &[u128],
    mul_cells: &[crate::worker::ScoreCell],
    sqr_cells: &[crate::worker::ScoreCell],
    div_cells: &[crate::worker::ScoreCell],
) {
    let arithmetic_cells = mul_cells.len().saturating_add(sqr_cells.len());
    for ((&tuned_time, &baseline_time), cell) in tuned
        .iter()
        .zip(baseline)
        .take(arithmetic_cells)
        .zip(mul_cells.iter().chain(sqr_cells))
    {
        let ratio = tuned_time
            .saturating_mul(1_000)
            .div_euclid(baseline_time.max(1));
        let verdict = if ratio < 1_000 {
            format!("tuned {ratio} per-mille (faster)")
        } else {
            format!("tuned {ratio} per-mille (slower)")
        };
        println!("  {}x{} limbs: {verdict}", cell.len_a, cell.len_b);
    }
    for ((&tuned_time, &baseline_time), cell) in tuned
        .iter()
        .zip(baseline)
        .skip(arithmetic_cells)
        .zip(div_cells)
    {
        let ratio = tuned_time
            .saturating_mul(1_000)
            .div_euclid(baseline_time.max(1));
        let verdict = if ratio < 1_000 {
            format!("tuned {ratio} per-mille (faster)")
        } else {
            format!("tuned {ratio} per-mille (slower)")
        };
        println!("  division {}x{} limbs: {verdict}", cell.len_a, cell.len_b);
    }
}

const FORMAT_VALIDATION_ITERATIONS: u32 = 5_000;

pub fn report_formatting_boundaries(session: &mut TuneSession) {
    println!("\nFormatting boundary report");
    for &radix in &[3, 5, 9, 10, 11, 19, 36] {
        let threshold = match radix {
            3..=9 => session.profile.radix_small_recursive,
            10 => session.profile.radix_decimal_recursive,
            _ => session.profile.radix_large_recursive,
        };
        if threshold >= usize::MAX - 16 {
            println!("  Radix {radix:02}: recursive formatting disabled");
            continue;
        }
        let lengths = [
            threshold.saturating_sub(1).max(4),
            threshold,
            threshold.saturating_add(4),
            threshold.checked_mul(2).unwrap_or(threshold),
        ];
        let mut prev = 0;
        for len in lengths {
            if len == prev {
                continue;
            }
            prev = len;
            let spec = crate::harness::FormattingPairSpec {
                baseline: "schoolbook",
                candidate: "recursive",
                radix,
                len,
                quality: crate::measure::ProbeQuality::Precise,
                iterations: FORMAT_VALIDATION_ITERATIONS,
            };
            if let Some((baseline_time, candidate_time)) = session
                .harness
                .score_formatting_pair(&session.profile, spec)
            {
                let winner = if candidate_time < baseline_time {
                    "recursive"
                } else {
                    "schoolbook"
                };
                println!(
                    "  Radix {radix:02} | len {len:4}: schoolbook {baseline_time:8} ns, recursive {candidate_time:8} ns -> {winner} wins"
                );
            }
        }
    }
}
