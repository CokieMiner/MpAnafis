//! Simultaneous acceptance and rejection rules for scheduled confirmation looks.

use crate::measure::RatioEstimate;

/// One weighted ratio bound. Zero weights exclude cells from this objective.
#[derive(Clone, Debug)]
pub struct ScoreLimit {
    pub weights: Vec<u32>,
    pub maximum: u128,
}

/// All objectives and individual cells must pass before accepting a profile.
#[derive(Clone, Debug)]
pub struct ComparisonRule {
    pub scores: Vec<ScoreLimit>,
    pub cell_maximum: u128,
}

/// Scheduled bounds accept the complete rule, reject a limit, or request more data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonDecision {
    Accept,
    Reject,
    Unresolved,
}

impl ComparisonRule {
    /// Evaluate simultaneous median bounds without treating an unresolved
    /// interval as a point estimate. Lower sums round down, upper sums round up.
    /// Invalid rules and arithmetic overflow cannot establish acceptance.
    pub fn decision(&self, estimates: &[RatioEstimate]) -> ComparisonDecision {
        if estimates.is_empty()
            || self.scores.is_empty()
            || estimates.iter().any(|estimate| {
                estimate.median == 0
                    || estimate.lower > estimate.median
                    || estimate.median > estimate.upper
            })
        {
            return ComparisonDecision::Unresolved;
        }
        if estimates
            .iter()
            .any(|estimate| estimate.lower > self.cell_maximum)
        {
            return ComparisonDecision::Reject;
        }
        let mut accepted = estimates
            .iter()
            .all(|estimate| estimate.upper <= self.cell_maximum);
        for score in &self.scores {
            if score.weights.len() != estimates.len() {
                return ComparisonDecision::Unresolved;
            }
            let mut total = 0_u128;
            let mut lower = Some(0_u128);
            let mut upper = Some(0_u128);
            for (estimate, &weight) in estimates.iter().zip(&score.weights) {
                if weight == 0 {
                    continue;
                }
                let wide_weight = u128::from(weight);
                let Some(next) = total.checked_add(wide_weight) else {
                    return ComparisonDecision::Unresolved;
                };
                total = next;
                lower =
                    lower.and_then(|sum| estimate.lower.checked_mul(wide_weight)?.checked_add(sum));
                upper =
                    upper.and_then(|sum| estimate.upper.checked_mul(wide_weight)?.checked_add(sum));
            }
            if total == 0 {
                return ComparisonDecision::Unresolved;
            }
            if lower.is_some_and(|sum| sum.div_euclid(total) > score.maximum) {
                return ComparisonDecision::Reject;
            }
            accepted &= upper.is_some_and(|sum| sum.div_ceil(total) <= score.maximum);
        }
        if accepted {
            ComparisonDecision::Accept
        } else {
            ComparisonDecision::Unresolved
        }
    }
}
