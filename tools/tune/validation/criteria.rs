//! Frozen aggregate, family, and individual-cell regression limits.

use core::ops::Range;

use super::{
    CandidateHarness, ComparisonRule, GCD_SCORE_CASES, GcdWorker, SCORE_SCALE, ScoreLimit,
};

/// Maximum aggregate regression tolerance allowed for timing jitter on identical code paths (0.5%).
pub const VALIDATION_AGGREGATE_TOLERANCE_PPM: u128 = 5_000;

/// Maximum individual family regression tolerance allowed for timing jitter (1.5%).
pub const VALIDATION_FAMILY_TOLERANCE_PPM: u128 = 15_000;

/// Per-cell jitter allowance (3%). An isolated size or shape regression cannot
/// be hidden by a faster operation or by the other cells in its family.
const VALIDATION_CELL_TOLERANCE_PPM: u128 = 30_000;

/// Production-dispatch validation and shared candidate regression guards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Validation;

impl Validation {
    /// Freeze the aggregate objective and complete-family guards before sampling.
    /// Families reuse the supplied logarithmic or equal cell weights; every
    /// individual cell also retains its independent regression limit.
    #[must_use]
    pub fn comparison_rule(
        weights: &[u32],
        maximum: u128,
        families: &[Range<usize>],
    ) -> ComparisonRule {
        let mut scores = vec![ScoreLimit {
            weights: weights.to_vec(),
            maximum,
        }];
        for family in families {
            let mut family_weights = vec![0; weights.len()];
            family_weights
                .get_mut(family.clone())
                .expect("family lies within catalog")
                .copy_from_slice(
                    weights
                        .get(family.clone())
                        .expect("family lies within catalog"),
                );
            if family_weights.iter().any(|&weight| weight > 0) {
                scores.push(ScoreLimit {
                    weights: family_weights,
                    maximum: SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM),
                });
            }
        }
        ComparisonRule {
            scores,
            cell_maximum: SCORE_SCALE.saturating_add(VALIDATION_CELL_TOLERANCE_PPM),
        }
    }

    /// Validate the complete family-major worker vector. The same guard rejects
    /// isolated regressions during candidate selection and fresh final A/B/B/A
    /// validation; malformed or zero-duration worker output also rejects a profile.
    #[must_use]
    pub fn gcd_non_regresses(tuned: &[u128], baseline: &[u128]) -> bool {
        let weights = GcdWorker::cell_weights(&GCD_SCORE_CASES);
        if !Self::cells_non_regress(tuned, baseline, weights.len()) {
            return false;
        }
        let family_weights = vec![1; GCD_SCORE_CASES.len()];
        tuned
            .chunks(GCD_SCORE_CASES.len())
            .zip(baseline.chunks(GCD_SCORE_CASES.len()))
            .all(|(candidate, reference)| {
                CandidateHarness::relative_score(candidate, reference, &family_weights)
                    <= SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM)
            })
    }

    /// Reject malformed vectors and isolated size/shape regressions before
    /// aggregate scoring can hide them. Zero durations never represent a sample.
    #[must_use]
    pub fn cells_non_regress(tuned: &[u128], baseline: &[u128], count: usize) -> bool {
        if count == 0 || tuned.len() != count || baseline.len() != count {
            return false;
        }
        let limit = SCORE_SCALE.saturating_add(VALIDATION_CELL_TOLERANCE_PPM);
        tuned.iter().zip(baseline).all(|(&candidate, &reference)| {
            candidate != 0
                && reference != 0
                && candidate
                    .checked_mul(SCORE_SCALE)
                    .zip(reference.checked_mul(limit))
                    .is_some_and(|(scaled_candidate, scaled_reference)| {
                        scaled_candidate <= scaled_reference
                    })
        })
    }
}
