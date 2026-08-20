//! Stateful host-tuning context shared by every measurement phase.
//!
//! A session is the ownership boundary for one candidate profile search. It
//! keeps the candidate and immutable architecture defaults together with the
//! measurement margin, cache-backed worker harness, report decisions, and
//! machine metadata. Phase modules therefore receive one state object instead
//! of independently threading five correlated arguments.

use std::path::PathBuf;

use crate::{
    crossovers::Calibration, harness::CandidateHarness, measure::acceptance_margin, store,
    tuning_profile::TuningProfile,
};

/// Complete state for one host autotuning run.
pub struct TuneSession {
    /// Candidate profile mutated by tuning phases.
    pub profile: TuningProfile,
    /// Architecture profile used as the end-to-end validation baseline.
    pub defaults: TuningProfile,
    /// Measured host noise and timing-bucket context.
    pub calibration: Calibration,
    /// Acceptance margin derived from [`Calibration::noise_cv`], in ppm.
    pub margin_ppm: u32,
    /// Persistent worker score cache and rebuild candidate file.
    pub harness: CandidateHarness,
    /// Ordered profile decisions written to the machine report.
    pub decisions: Vec<(String, String)>,
    /// Stable machine-specific output directory.
    pub machine_dir: PathBuf,
    /// Human-readable CPU/core identity used in reports and output headers.
    pub cpu: String,
    /// ISO date associated with this run.
    pub date: String,
}

impl TuneSession {
    /// Construct a session after host calibration and machine identity have
    /// been collected. The margin and score cache are established exactly once.
    #[must_use]
    pub fn new(
        defaults: &TuningProfile,
        calibration: Calibration,
        machine_dir: PathBuf,
        cpu: String,
        date: String,
    ) -> Self {
        let margin_ppm = acceptance_margin(calibration.noise_cv);
        let harness = CandidateHarness::new(
            &machine_dir.join(store::SCORE_CACHE_NAME),
            calibration.timing_bucket_ms,
        );
        Self {
            profile: *defaults,
            defaults: *defaults,
            calibration,
            margin_ppm,
            harness,
            decisions: Vec::new(),
            machine_dir,
            cpu,
            date,
        }
    }

    /// Record a decision in the report log while preserving phase order.
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.decisions.push((name.into(), value.into()));
    }
}

#[cfg(test)]
mod tests {
    use super::TuneSession;
    use crate::{crossovers::Calibration, tuning_profile::TuningProfile};

    #[test]
    fn session_starts_with_an_independent_candidate_copy() {
        let defaults = TuningProfile::default();
        let session = TuneSession::new(
            &defaults,
            Calibration {
                noise_cv: 0.01,
                timing_bucket_ms: 2,
            },
            std::env::temp_dir().join("mp-anafis-session-test"),
            "test-cpu".to_owned(),
            "2026-01-01".to_owned(),
        );
        assert_eq!(session.profile, session.defaults);
        assert_eq!(session.margin_ppm, 30_000);
        assert!(session.decisions.is_empty());
    }

    #[test]
    fn record_preserves_order_and_owns_text() {
        let defaults = TuningProfile::default();
        let mut session = TuneSession::new(
            &defaults,
            Calibration {
                noise_cv: 0.01,
                timing_bucket_ms: 2,
            },
            std::env::temp_dir().join("mp-anafis-session-test"),
            "test-cpu".to_owned(),
            "2026-01-01".to_owned(),
        );
        session.record("first", "one");
        session.record(String::from("second"), String::from("two"));
        assert_eq!(
            session.decisions,
            [
                ("first".to_owned(), "one".to_owned()),
                ("second".to_owned(), "two".to_owned())
            ]
        );
    }
}
