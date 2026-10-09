//! Formatting correctness at crossovers and production parsing regression gates.

use crate::worker::{CONSUMER_FORMAT_RADICES, PARSING_RADICES};

use super::{
    CandidateHarness, CompiledTuner, FormattingPairSpec, ProbeQuality, SCORE_SCALE, ScoreDomain,
    TuneSession, VALIDATION_AGGREGATE_TOLERANCE_PPM, Validation,
};

const FORMAT_VALIDATION_ITERATIONS: u32 = 5_000;

impl Validation {
    /// Verify that both formatting tiers agree and can be measured at every tuned
    /// threshold boundary. A missing worker result rejects installation.
    pub fn formatting_boundaries(session: &mut TuneSession) -> bool {
        println!("\nFormatting boundary validation");
        let mut valid = true;
        for radix in CONSUMER_FORMAT_RADICES {
            let threshold = match radix {
                3..=9 => session.profile.radix_format_small_recursive,
                10 => session.profile.radix_format_decimal_recursive,
                _ => session.profile.radix_format_large_recursive,
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
                let spec = FormattingPairSpec {
                    baseline: "schoolbook",
                    candidate: "recursive",
                    radix,
                    len,
                    quality: ProbeQuality::Precise,
                    iterations: FORMAT_VALIDATION_ITERATIONS,
                };
                if let Some((baseline_time, candidate_time, upper_ratio)) = session
                    .harness
                    .score_formatting_pair(&session.profile, spec)
                {
                    let winner = if upper_ratio < SCORE_SCALE {
                        "recursive confirmed"
                    } else {
                        "no confirmed recursive advantage"
                    };
                    println!(
                        "  Radix {radix:02} | len {len:4}: schoolbook {baseline_time:8} ns, recursive {candidate_time:8} ns, upper ratio {upper_ratio} ppm -> {winner}"
                    );
                } else {
                    valid = false;
                    println!("  Radix {radix:02} | len {len:4}: validation worker failed");
                }
            }
        }
        session.record(
            "FORMATTING_BOUNDARY_VALIDATION".to_owned(),
            if valid {
                "passed".to_owned()
            } else {
                "rejected: one or more boundary workers failed".to_owned()
            },
        );
        valid
    }

    /// Compare complete parsing calls at the default and selected root/leaf
    /// neighbours. Each radix remains a separate family regression guard.
    pub fn parsing_profiles(session: &mut TuneSession) -> bool {
        session.harness.parsing_chunks =
            CompiledTuner::parsing_grid(&[session.defaults, session.profile], false);
        session.harness.selected_cells.clear();
        let weights = ScoreDomain::Parsing.cell_weights(&session.harness);
        let width_count = session.harness.parsing_chunks.len();
        let families: Vec<_> = (0..PARSING_RADICES.len())
            .map(|index| {
                let start = index
                    .checked_mul(width_count)
                    .expect("finite parsing catalog");
                start
                    ..start
                        .checked_add(width_count)
                        .expect("finite parsing catalog")
            })
            .collect();
        let rule = Self::comparison_rule(
            &weights,
            SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM),
            &families,
        );
        let Some(ratios) = session.harness.compare_profiles(
            &session.defaults,
            &session.profile,
            ScoreDomain::Parsing,
            &rule,
        ) else {
            session.record("PARSING_VALIDATION", "paired parsing execution failed");
            return false;
        };
        let baseline = vec![SCORE_SCALE; weights.len()];
        let valid = Self::cells_non_regress(&ratios, &baseline, weights.len())
            && rule.scores.iter().all(|score| {
                CandidateHarness::relative_score(&ratios, &baseline, &score.weights)
                    <= score.maximum
            });
        session.record(
            "PARSING_VALIDATION",
            format!(
                "accepted={valid}; chunks={:?}; upper_bounds_ppm={ratios:?}",
                session.harness.parsing_chunks,
            ),
        );
        valid
    }
}
