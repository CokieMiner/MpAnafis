//! Invalid measurements must never win a tuning search.

use core::time::Duration;
use std::{
    env::temp_dir,
    fs::{File, metadata, read_to_string, remove_dir_all, remove_file, write},
    path::PathBuf,
    process::id as process_id,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::measure::RatioEstimate;

use super::{
    CandidateHarness, ComparisonDecision, ComparisonRule, Executables, SCORE_SCALE, ScoreLimit,
    executable::CandidateFile,
};

#[test]
fn paired_ratios_preserve_cells_and_reject_missing_samples() {
    let reference = vec![vec![100, 200], vec![300, 600]];
    let candidate = vec![vec![50, 400], vec![150, 1_200]];
    assert_eq!(
        CandidateHarness::paired_ratios(&reference, &candidate),
        Some(vec![500_000, 2_000_000])
    );
    for malformed in [
        vec![],
        vec![vec![1]],
        vec![vec![1], vec![1, 2]],
        vec![vec![0], vec![1]],
        vec![vec![u128::MAX], vec![1]],
    ] {
        assert_eq!(
            CandidateHarness::paired_ratios(&reference, &malformed),
            None
        );
        assert_eq!(
            CandidateHarness::paired_ratios(&malformed, &reference),
            None
        );
    }
}

#[test]
fn paired_ratios_bound_the_exact_sum_ratio_without_mean_truncation() {
    for numerator in 2_u128..32 {
        for denominator in 2_u128..32 {
            let baseline = [vec![1], vec![denominator - 1]];
            let candidate = [vec![1], vec![numerator - 1]];
            let ratios = CandidateHarness::paired_ratios(&baseline, &candidate)
                .expect("positive bounded measurements");
            let scaled = *ratios.first().expect("one score cell") * denominator;
            let exact = numerator * SCORE_SCALE;
            assert!(scaled >= exact);
            assert!(scaled < exact + denominator);
        }
    }
    let baseline = [vec![1], vec![1]];
    for invalid in [[vec![0], vec![1]], [vec![u128::MAX], vec![1]]] {
        assert_eq!(CandidateHarness::paired_ratios(&baseline, &invalid), None);
        assert_eq!(CandidateHarness::paired_ratios(&invalid, &baseline), None);
    }
}

#[test]
fn cargo_artifact_paths_preserve_escaping_and_reject_ambiguity() {
    for (message, expected) in [
        (r#"{"executable":"/tmp/a b/mp-tune"}"#, "/tmp/a b/mp-tune"),
        (
            r#"{"executable":"C:\\work\\mp-tune.exe"}"#,
            "C:\\work\\mp-tune.exe",
        ),
        (r#"{"executable":"/tmp/\u0061/mp-tune"}"#, "/tmp/a/mp-tune"),
    ] {
        assert_eq!(
            Executables::cargo_executable(message),
            Some(PathBuf::from(expected))
        );
    }
    for message in [
        "",
        r#"{"executable":null}"#,
        r#"{"executable":"unterminated}"#,
        "{\"executable\":\"a\"}\n{\"executable\":\"b\"}",
    ] {
        assert_eq!(Executables::cargo_executable(message), None);
    }
}

#[test]
fn unchanged_profiles_preserve_the_build_input_timestamp() {
    let candidate = CandidateFile::new().expect("temporary profile");
    Executables::write_candidate_source(&candidate.path, "first profile").expect("write candidate");
    let old_time = UNIX_EPOCH + Duration::from_secs(60);
    File::options()
        .write(true)
        .open(&candidate.path)
        .expect("candidate file")
        .set_modified(old_time)
        .expect("set fixed old timestamp");
    Executables::write_candidate_source(&candidate.path, "first profile").expect("reuse candidate");
    assert_eq!(
        metadata(&candidate.path)
            .expect("candidate metadata")
            .modified()
            .expect("modified time"),
        old_time
    );
    Executables::write_candidate_source(&candidate.path, "second profile")
        .expect("replace candidate");
    assert_eq!(
        read_to_string(&candidate.path).expect("candidate source"),
        "second profile"
    );
    assert_ne!(
        metadata(&candidate.path)
            .expect("candidate metadata")
            .modified()
            .expect("modified time"),
        old_time
    );
}

#[test]
fn relative_scores_reject_empty_zero_and_overflowing_measurements() {
    for (samples, baseline, weights) in [
        (&[][..], &[][..], &[][..]),
        (&[1][..], &[1][..], &[0][..]),
        (&[0][..], &[1][..], &[1][..]),
        (&[1][..], &[0][..], &[1][..]),
        (&[1][..], &[1, 2][..], &[1][..]),
        (&[u128::MAX][..], &[u128::MAX][..], &[1][..]),
    ] {
        assert_eq!(
            CandidateHarness::relative_score(samples, baseline, weights),
            u128::MAX
        );
    }
    assert_eq!(
        CandidateHarness::relative_score(&[100, 200], &[100, 100], &[3, 1]),
        1_250_000
    );
    assert_eq!(
        CandidateHarness::relative_score(&[100, 200], &[100, 100], &[1, 0]),
        SCORE_SCALE
    );
}

#[test]
fn weighted_confirmation_bounds_round_up_at_fractional_ppm() {
    assert_eq!(CandidateHarness::relative_score(&[1], &[3], &[1]), 333_334);
    assert_eq!(
        CandidateHarness::relative_score(
            &[SCORE_SCALE + 1, SCORE_SCALE],
            &[SCORE_SCALE, SCORE_SCALE],
            &[1, 1],
        ),
        SCORE_SCALE + 1
    );
}

#[test]
fn worker_protocol_rejects_ambiguous_or_invalid_scores() {
    assert_eq!(
        Executables::parse_measurements("log\nSCORE=10,20\n", "SCORE="),
        Some(vec![10, 20])
    );
    for source in [
        "",
        "SCORE=",
        "SCORE=0",
        "SCORE=-1",
        "SCORE=10,",
        "SCORE=1\nSCORE=2",
    ] {
        assert_eq!(
            Executables::parse_measurements(source, "SCORE="),
            None,
            "{source}"
        );
    }
}

#[test]
fn simultaneous_candidate_files_are_independent_and_removed_on_drop() {
    let first = CandidateFile::new().expect("temporary profile");
    let second = CandidateFile::new().expect("temporary profile");
    assert_ne!(first.path, second.path);
    assert!(first.path.is_file());
    let path = first.path.clone();
    drop(first);
    assert!(!path.exists());
    assert!(second.path.is_file());
}

#[test]
fn harness_construction_propagates_artifact_directory_errors() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-harness-blocked-{}-{nonce}", process_id()));
    write(&directory, "existing data").expect("blocking parent file");
    let result = CandidateHarness::new(&directory.join("scores.json"), 0);
    assert!(result.is_err());
    assert_eq!(
        read_to_string(&directory).expect("preserved data"),
        "existing data"
    );
    remove_file(directory).expect("test cleanup");
}

#[test]
fn context_failure_remains_latched_after_the_original_identity_returns() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-harness-context-{}-{nonce}", process_id()));
    let mut harness = CandidateHarness::new(&directory.join("scores.json"), 0)
        .expect("readable workspace and artifact directory");
    assert!(harness.check_context());
    let original = harness.source_context_hash;
    harness.source_context_hash ^= 1;
    assert!(!harness.check_context());
    harness.source_context_hash = original;
    assert!(!harness.check_context());
    assert!(harness.failed);
    drop(harness);
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn scheduled_rules_require_all_objectives_and_individual_cells() {
    let rule = ComparisonRule {
        scores: vec![
            ScoreLimit {
                weights: vec![1, 1],
                maximum: 990_000,
            },
            ScoreLimit {
                weights: vec![0, 1],
                maximum: 1_015_000,
            },
        ],
        cell_maximum: 1_030_000,
    };
    let fast = RatioEstimate {
        median: 900_000,
        lower: 890_000,
        upper: 910_000,
    };
    let stable = RatioEstimate {
        median: 1_000_000,
        lower: 990_000,
        upper: 1_010_000,
    };
    assert_eq!(rule.decision(&[fast, stable]), ComparisonDecision::Accept);
    let uncertain = RatioEstimate {
        median: 900_000,
        lower: 890_000,
        upper: 1_020_000,
    };
    assert_eq!(
        rule.decision(&[uncertain, stable]),
        ComparisonDecision::Unresolved
    );
    let family_regression = RatioEstimate {
        median: 1_020_000,
        lower: 1_016_000,
        upper: 1_025_000,
    };
    assert_eq!(
        rule.decision(&[fast, family_regression]),
        ComparisonDecision::Reject
    );
    let cell_regression = RatioEstimate {
        median: 1_040_000,
        lower: 1_031_000,
        upper: 1_050_000,
    };
    let aggregate_only = ComparisonRule {
        scores: vec![ScoreLimit {
            weights: vec![1, 0],
            maximum: 990_000,
        }],
        cell_maximum: 1_030_000,
    };
    assert_eq!(
        aggregate_only.decision(&[fast, cell_regression]),
        ComparisonDecision::Reject
    );
}

#[test]
fn scheduled_rules_round_upper_bounds_up_and_cannot_accept_invalid_data() {
    let rule = ComparisonRule {
        scores: vec![ScoreLimit {
            weights: vec![1, 1],
            maximum: 100,
        }],
        cell_maximum: 200,
    };
    let low = RatioEstimate {
        median: 100,
        lower: 100,
        upper: 100,
    };
    let high = RatioEstimate {
        median: 101,
        lower: 101,
        upper: 101,
    };
    assert_eq!(rule.decision(&[low, high]), ComparisonDecision::Unresolved);
    assert_eq!(rule.decision(&[low, low]), ComparisonDecision::Accept);
    assert_eq!(rule.decision(&[]), ComparisonDecision::Unresolved);
    assert_eq!(rule.decision(&[low]), ComparisonDecision::Unresolved);
    for invalid in [
        RatioEstimate {
            median: 0,
            lower: 0,
            upper: 0,
        },
        RatioEstimate {
            median: 100,
            lower: 101,
            upper: 100,
        },
        RatioEstimate {
            median: 101,
            lower: 100,
            upper: 100,
        },
    ] {
        assert_eq!(
            rule.decision(&[invalid, low]),
            ComparisonDecision::Unresolved
        );
    }
    for weights in [vec![0, 0], vec![2, 2]] {
        let invalid = ComparisonRule {
            scores: vec![ScoreLimit {
                weights,
                maximum: u128::MAX,
            }],
            cell_maximum: u128::MAX,
        };
        let enormous = RatioEstimate {
            median: u128::MAX,
            lower: u128::MAX,
            upper: u128::MAX,
        };
        assert_ne!(
            invalid.decision(&[enormous, enormous]),
            ComparisonDecision::Accept
        );
    }
}
