//! Tower failures are transactional; unavailable timings do not disable tiers.

use std::env::temp_dir;

use crate::crossovers::Calibration;

use super::{
    Candidate, FormattingTuner, Parameter, TierTuner, TowerWalker, TuneSession, TuningProfile,
};

#[test]
fn formatting_worker_failure_preserves_the_profile_without_panicking() {
    let defaults = TuningProfile::default();
    let mut session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 10_000,
            timing_bucket_ms: 2,
        },
        temp_dir().join("mp-anafis-formatting-failure-test"),
        "test-cpu".to_owned(),
        "2026-09-23".to_owned(),
    )
    .expect("session artifacts can be created");
    session.harness.reject("injected unavailable worker");
    FormattingTuner::tune_formatting(&mut session);
    assert_eq!(session.profile, defaults);
    assert!(!session.harness.check_context());
}

#[test]
fn failed_tower_restores_all_prior_threshold_updates() {
    let defaults = TuningProfile::default();
    let mut session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 10_000,
            timing_bucket_ms: 2,
        },
        temp_dir().join("mp-anafis-tower-test"),
        "test-cpu".to_owned(),
        "2026-09-23".to_owned(),
    )
    .expect("session artifacts can be created");
    let candidates = [
        Candidate {
            algo: 1,
            sizes: &[8, 16, 32],
            min_next_start: 8,
        },
        Candidate {
            algo: 2,
            sizes: &[32, 64, 128],
            min_next_start: 32,
        },
    ];
    let scoring_profile = session.profile;
    let thresholds = TowerWalker::tune_tower(
        &mut session,
        &scoring_profile,
        0,
        &candidates,
        |_session, _profile, _baseline, candidate, _start, _sizes, _tag, _iterations| {
            if candidate == 1 {
                Ok(Some(16))
            } else {
                Err("injected worker failure".to_owned())
            }
        },
        |state, _index, threshold| state.profile.karatsuba = threshold,
    );
    assert_eq!(thresholds, Vec::<usize>::new());
    assert_eq!(session.profile, defaults);
    assert!(!session.harness.check_context());
    assert!(
        session
            .decisions
            .iter()
            .any(|(_, decision)| decision.contains("tower restored"))
    );
}

#[test]
fn unconfirmed_tower_restores_the_input_profile() {
    let defaults = TuningProfile::default();
    let mut session = TuneSession::new(
        &defaults,
        Calibration {
            noise_cv_ppm: 10_000,
            timing_bucket_ms: 2,
        },
        temp_dir().join("mp-anafis-tower-inconclusive-test"),
        "test-cpu".to_owned(),
        "2026-10-01".to_owned(),
    )
    .expect("session artifacts can be created");
    let candidates = [
        Candidate {
            algo: 1,
            sizes: &[8, 16, 32],
            min_next_start: 8,
        },
        Candidate {
            algo: 2,
            sizes: &[32, 64, 128],
            min_next_start: 32,
        },
    ];
    let scoring_profile = session.profile;
    let thresholds = TowerWalker::tune_tower(
        &mut session,
        &scoring_profile,
        0,
        &candidates,
        |_session, _profile, _baseline, candidate, _start, _sizes, _tag, _iterations| {
            Ok((candidate == 1).then_some(16))
        },
        |state, _index, threshold| state.profile.karatsuba = threshold,
    );
    assert_eq!(thresholds, Vec::<usize>::new());
    assert_eq!(session.profile, defaults);
    assert!(session.harness.check_context());
    assert!(
        session
            .decisions
            .iter()
            .any(|(_, decision)| decision.contains("retained"))
    );
}

#[test]
fn recording_multiplication_tower_preserves_ssa_shadowing() {
    check_recorded_tower(false);
}

#[test]
fn recording_squaring_tower_preserves_ssa_shadowing() {
    check_recorded_tower(true);
}

fn check_recorded_tower(squaring: bool) {
    let defaults = TuningProfile::default();
    let (transform, fields, apply): (_, _, fn(&mut TuneSession, usize, usize)) = if squaring {
        (
            defaults.sqr_ssa,
            [
                Parameter::SQR_KARATSUBA_THRESHOLD,
                Parameter::SQR_TOOM_COOK_THRESHOLD,
                Parameter::SQR_TOOM_COOK_4_THRESHOLD,
                Parameter::SQR_TOOM_COOK_6_THRESHOLD,
                Parameter::SQR_TOOM_COOK_85_THRESHOLD,
            ],
            TierTuner::apply_squaring_threshold,
        )
    } else {
        (
            defaults.ssa,
            [
                Parameter::KARATSUBA_THRESHOLD,
                Parameter::TOOM_COOK_THRESHOLD,
                Parameter::TOOM_COOK_4_THRESHOLD,
                Parameter::TOOM_COOK_6_THRESHOLD,
                Parameter::TOOM_COOK_85_THRESHOLD,
            ],
            TierTuner::apply_multiplication_threshold,
        )
    };
    for crossover in [
        transform
            .checked_sub(1)
            .expect("positive transform threshold"),
        transform,
        transform
            .checked_add(1)
            .expect("finite transform threshold"),
    ] {
        let mut session = TuneSession::new(
            &defaults,
            Calibration {
                noise_cv_ppm: 10_000,
                timing_bucket_ms: 2,
            },
            temp_dir().join(format!("mp-anafis-tower-record-{squaring}-{crossover}")),
            "test-cpu".to_owned(),
            "2026-09-26".to_owned(),
        )
        .expect("session artifacts can be created");
        let candidates = [18, 259, 313, 1_000, crossover].map(|threshold| Candidate {
            algo: threshold,
            sizes: &[8, 16, 32, 256, 512, 1_024, 4_096],
            min_next_start: 8,
        });
        let scoring_profile = session.profile;
        let measured = TowerWalker::tune_tower(
            &mut session,
            &scoring_profile,
            0,
            &candidates,
            |_session, profile, _baseline, candidate, _start, _sizes, _tag, _iterations| {
                profile.validate().map_err(str::to_owned)?;
                Ok(Some(candidate))
            },
            apply,
        );
        assert_eq!(measured.last(), Some(&crossover));
        let installed = session.profile;
        assert!(installed.validate().is_ok());
        TowerWalker::record_thresholds(&mut session, &fields);
        assert_eq!(
            session.profile, installed,
            "recording must preserve the reconciled profile"
        );
        assert!(
            session.profile.validate().is_ok(),
            "the next phase must accept this profile"
        );
        let recorded = session
            .decisions
            .last()
            .expect("the final tier was recorded");
        assert_eq!(
            recorded.0,
            fields.last().expect("five recorded thresholds").name
        );
        assert_eq!(
            recorded.1,
            if crossover < transform {
                crossover
            } else {
                usize::MAX.checked_sub(1).expect("sentinel fits")
            }
            .to_string(),
            "the decision must describe the installed threshold"
        );
    }
}
