//! Production arithmetic validation against the reference profile.

use super::{
    CandidateHarness, ComparisonRule, CompiledTuner, DivisionWorker, GCD_SCORE_CASES, GcdWorker,
    PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, SCORE_SCALE, ScoreCell, ScoreDomain, TuneSession,
    VALIDATION_AGGREGATE_TOLERANCE_PPM, VALIDATION_FAMILY_TOLERANCE_PPM, Validation,
};

impl Validation {
    /// Score the tuned profile against the architecture defaults on the
    /// production dispatcher. Returns `true` when the tuned profile verifies
    /// non-regression against the defaults within timing jitter tolerances, and
    /// records the per-cell verdicts in `decisions`.
    pub fn end_to_end(session: &mut TuneSession) -> bool {
        println!("\nEnd-to-end validation (production dispatcher)");
        session.harness.division_grid =
            CompiledTuner::division_grid(&[session.defaults, session.profile], false);
        session.harness.selected_cells.clear();
        let div_weights =
            DivisionWorker::cell_weights(&DivisionWorker::cases(&session.harness.division_grid));
        let (weights, rule) = Self::production_rule(&div_weights);
        let Some(tuned) = session.harness.compare_profiles(
            &session.defaults,
            &session.profile,
            ScoreDomain::Production,
            &rule,
        ) else {
            println!("Validation failed: paired production execution failed");
            return false;
        };
        let baseline = vec![SCORE_SCALE; weights.len()];
        session.record(
            "END_TO_END_SAMPLES",
            format!(
                "ABBA/BAAB median-ratio upper bounds_ppm={}",
                tuned
                    .iter()
                    .map(u128::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        );
        let tuned_score = CandidateHarness::relative_score(&tuned, &baseline, &weights);
        let baseline_score = SCORE_SCALE;
        let mul_weights = ScoreCell::cell_weights(&PRODUCTION_MUL_CELLS, &[]);
        let sqr_weights = ScoreCell::cell_weights(&[], &PRODUCTION_SQR_CELLS);
        let mul_score = segment_score(&tuned, &baseline, 0, &mul_weights);
        let sqr_offset = PRODUCTION_MUL_CELLS.len();
        let sqr_score = segment_score(&tuned, &baseline, sqr_offset, &sqr_weights);
        let div_offset = sqr_offset.saturating_add(PRODUCTION_SQR_CELLS.len());
        let div_score = segment_score(&tuned, &baseline, div_offset, &div_weights);
        let gcd_offset = div_offset.saturating_add(div_weights.len());
        report_cells(
            &tuned,
            &baseline,
            &PRODUCTION_MUL_CELLS,
            &PRODUCTION_SQR_CELLS,
        );

        let aggregate_limit = SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM);
        let aggregate_non_regresses = tuned_score <= aggregate_limit;
        let family_limit = SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM);
        let no_family_regresses = [mul_score, sqr_score, div_score]
            .into_iter()
            .all(|score| score <= family_limit);
        println!(
            "  family upper bounds: multiplication {mul_score} ppm, square {sqr_score} ppm, division {div_score} ppm (family limit: {family_limit} ppm)"
        );
        let gcd_tuned = tuned.get(gcd_offset..).unwrap_or_default();
        let gcd_baseline = baseline.get(gcd_offset..).unwrap_or_default();
        let no_gcd_regression = Self::gcd_non_regresses(gcd_tuned, gcd_baseline);
        let no_cell_regression = Self::cells_non_regress(&tuned, &baseline, weights.len());
        let consumers_valid = Self::consumer_profiles(session);
        let parsing_valid = Self::parsing_profiles(session);
        let wins = consumers_valid
            && parsing_valid
            && aggregate_non_regresses
            && no_family_regresses
            && no_gcd_regression
            && no_cell_regression
            && Self::holdout_profiles(session);
        if wins {
            println!(
                "Validation passed: tuned profile {tuned_score} ppm vs defaults {baseline_score} (aggregate non-regression limit {aggregate_limit} ppm)"
            );
            session.record(
                "END_TO_END_VALIDATION".to_owned(),
                format!("passed at {tuned_score} ppm vs defaults {baseline_score}"),
            );
        } else {
            println!(
                "Validation REJECTED: tuned profile {tuned_score} ppm vs defaults {baseline_score} \
             (exceeded arithmetic, consumer, parsing, or held-out limits); \
             not installing the tuned profile"
            );
            session.record(
                "END_TO_END_VALIDATION".to_owned(),
                format!("rejected: {tuned_score} ppm, defaults {baseline_score}"),
            );
        }
        wins
    }

    /// Partition the production vector into multiplication, square, division,
    /// and four complete GCD-family objectives before confirmation begins.
    #[must_use]
    pub fn production_rule(div_weights: &[u32]) -> (Vec<u32>, ComparisonRule) {
        let mut weights = ScoreCell::cell_weights(&PRODUCTION_MUL_CELLS, &PRODUCTION_SQR_CELLS);
        weights.extend_from_slice(div_weights);
        weights.extend(GcdWorker::cell_weights(&GCD_SCORE_CASES));
        let counts = [
            PRODUCTION_MUL_CELLS.len(),
            PRODUCTION_SQR_CELLS.len(),
            div_weights.len(),
            GCD_SCORE_CASES.len(),
            GCD_SCORE_CASES.len(),
            GCD_SCORE_CASES.len(),
            GCD_SCORE_CASES.len(),
        ];
        let mut offset = 0_usize;
        let families: Vec<_> = counts
            .into_iter()
            .map(|count| {
                let start = offset;
                offset = offset.checked_add(count).expect("production catalog fits");
                start..offset
            })
            .collect();
        let rule = Self::comparison_rule(
            &weights,
            SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM),
            &families,
        );
        (weights, rule)
    }
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
    CandidateHarness::relative_score(tuned_segment, baseline_segment, weights)
}

fn report_cells(
    tuned: &[u128],
    baseline: &[u128],
    mul_cells: &[ScoreCell],
    sqr_cells: &[ScoreCell],
) {
    let arithmetic_cells = mul_cells.len().saturating_add(sqr_cells.len());
    for ((&tuned_bound, &baseline_ratio), cell) in tuned
        .iter()
        .zip(baseline)
        .take(arithmetic_cells)
        .zip(mul_cells.iter().chain(sqr_cells))
    {
        let ratio = tuned_bound
            .saturating_mul(1_000)
            .div_euclid(baseline_ratio.max(1));
        let verdict = if ratio < 1_000 {
            format!("median-ratio upper bound {ratio} per-mille (advantage confirmed)")
        } else {
            format!("median-ratio upper bound {ratio} per-mille (advantage unresolved)")
        };
        println!("  {}x{} limbs: {verdict}", cell.len_a, cell.len_b);
    }
}
