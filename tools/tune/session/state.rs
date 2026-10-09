//! Complete host-tuning session state.

use std::path::PathBuf;

use super::{
    Calibration, CandidateHarness, CrossoverMeasure, SCORE_CACHE_NAME, SCORE_SCALE, TuningProfile,
};

/// Complete state for one host autotuning run.
#[derive(Debug)]
pub struct TuneSession {
    /// Candidate profile mutated by tuning phases.
    pub profile: TuningProfile,
    /// Architecture profile used as the end-to-end validation baseline.
    pub defaults: TuningProfile,
    /// Measured host noise and timing-bucket context.
    pub calibration: Calibration,
    /// Acceptance margin derived from [`Calibration::noise_cv_ppm`].
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
    /// Whether every formatting boundary probe completed its correctness
    /// comparison before installation validation.
    pub formatting_validated: bool,
}

impl TuneSession {
    /// Construct a session after host calibration and machine identity have
    /// been collected. The margin and score cache are established exactly once.
    ///
    /// # Errors
    /// Returns an error if the source context or candidate artifact directory
    /// cannot be established.
    pub fn new(
        defaults: &TuningProfile,
        calibration: Calibration,
        machine_dir: PathBuf,
        cpu: String,
        date: String,
    ) -> Result<Self, String> {
        let margin_ppm = CrossoverMeasure::acceptance_margin(calibration.noise_cv_ppm);
        let mut harness = CandidateHarness::new(
            &machine_dir.join(SCORE_CACHE_NAME),
            calibration.timing_bucket_ms,
        )?;
        harness.acceptance_limit = SCORE_SCALE
            .checked_sub(u128::from(margin_ppm))
            .and_then(|limit| limit.checked_sub(1))
            .expect("calibrated margin is below the score scale");
        Ok(Self {
            profile: *defaults,
            defaults: *defaults,
            calibration,
            margin_ppm,
            harness,
            decisions: Vec::new(),
            machine_dir,
            cpu,
            date,
            formatting_validated: false,
        })
    }

    /// Record a decision in the report log while preserving phase order.
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.decisions.push((name.into(), value.into()));
    }
}
