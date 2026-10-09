//! Tests for tuning session state and decision recording.

use std::env::temp_dir;

use super::{Calibration, TuneSession, TuningProfile};

#[test]
fn session_starts_with_an_independent_candidate_copy() {
    let defaults = TuningProfile::default();
    let session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 10_000,
            timing_bucket_ms: 2,
        },
        temp_dir().join("mp-anafis-session-test"),
        "test-cpu".to_owned(),
        "2026-01-01".to_owned(),
    )
    .expect("session artifacts can be created");
    assert_eq!(session.profile, session.defaults);
    assert_eq!(session.margin_ppm, 10_500);
    assert_eq!(session.decisions, Vec::new());
}

#[test]
fn record_preserves_order_and_owns_text() {
    let defaults = TuningProfile::default();
    let mut session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 10_000,
            timing_bucket_ms: 2,
        },
        temp_dir().join("mp-anafis-session-test"),
        "test-cpu".to_owned(),
        "2026-01-01".to_owned(),
    )
    .expect("session artifacts can be created");
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
