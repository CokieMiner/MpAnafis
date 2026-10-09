//! Screening-cache, context-identity, and report persistence tests.

use std::{
    collections::HashMap,
    env::temp_dir,
    fs::{create_dir_all, read_dir, read_to_string, remove_dir_all, write},
    io::ErrorKind,
    process::id as process_id,
    thread::scope,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{MeasurementContext, ScoreStore, TuningProfile};

#[test]
fn score_cache_round_trips() {
    let mut entries = HashMap::new();
    drop(entries.insert(1_u64, vec![10, 20, 30]));
    drop(entries.insert(u64::MAX, vec![378_125, 2_174_824_933]));
    let encoded = ScoreStore::encode_scores(&entries);
    let parsed = ScoreStore::parse_scores(&encoded).expect("round trip must parse");
    assert_eq!(parsed, entries);
}

#[test]
fn fnv1a_matches_the_reference_vector() {
    assert_eq!(ScoreStore::fnv1a(b""), 0xCBF2_9CE4_8422_2325);
    assert_eq!(ScoreStore::fnv1a(b"a"), 0xAF63_DC4C_8601_EC8C);
}

#[test]
fn profile_hash_is_deterministic_and_distinguishes_profiles() {
    let first = TuningProfile::portable();
    let second = TuningProfile::portable();
    assert_eq!(
        ScoreStore::profile_hash(&first),
        ScoreStore::profile_hash(&second)
    );
    let mut different = first;
    different.karatsuba = 99;
    assert_ne!(
        ScoreStore::profile_hash(&first),
        ScoreStore::profile_hash(&different)
    );
}

#[test]
fn machine_dir_sanitizes_the_cpu_name() {
    let dir = ScoreStore::machine_dir("AMD Ryzen 9 7950X (16 cores)");
    let rendered = dir.to_string_lossy();
    assert!(
        rendered.contains("AMD_Ryzen_9_7950X_16_cores_"),
        "{rendered}"
    );
    assert!(
        !rendered.contains('(') && !rendered.contains(' '),
        "{rendered}"
    );
}

#[test]
fn measurement_context_key_is_stable_and_single_line() {
    let context = MeasurementContext {
        platform: "model=test|cpu=3".to_owned(),
        target: "arch=x86_64".to_owned(),
        compiler: "rustc 1.90\ncommit".to_owned(),
        flags: "RUSTFLAGS=-C opt".to_owned(),
        executor: "built-in-thread-pool;workers=4".to_owned(),
    };
    let key = context.key();
    assert_eq!(key, context.key());
    assert!(key.contains("platform="));
    assert!(key.contains("\\|"));
    assert!(key.contains("\\n"));
    assert!(!key.contains('\n'));
    assert!(key.contains("executor="));
}

#[test]
fn report_strings_escape_every_json_control_character() {
    assert_eq!(
        ScoreStore::json_string("profile\n\t\u{07}\\\""),
        "\"profile\\n\\t\\u0007\\\\\\\"\""
    );
}

#[test]
fn report_publication_preserves_decisions_and_complete_profile_source() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-report-{}-{nonce}", process_id()));
    let path = directory.join("report.json");
    let profile = TuningProfile::portable();
    let decisions = [("knob\n\"".to_owned(), "accepted\t\\".to_owned())];
    ScoreStore::write_report(&path, "CPU\n\"", "2026-10-04", &profile, &decisions)
        .expect("complete report");
    let report = read_to_string(&path).expect("published report");
    assert!(report.starts_with("{\"cpu\":\"CPU\\n\\\"\",\"date\":\"2026-10-04\""));
    assert!(report.contains("\"knob\":\"knob\\n\\\"\",\"outcome\":\"accepted\\t\\\\\""));
    assert!(report.contains(&ScoreStore::json_string(
        &profile.render("// Final tuned profile")
    )));
    assert!(report.ends_with("}\n"));
    assert!(!path.with_extension("tmp").exists());
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn report_publication_propagates_directory_errors_and_preserves_existing_data() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-report-parent-{}-{nonce}", process_id()));
    create_dir_all(&directory).expect("test directory");
    let parent = directory.join("blocked");
    write(&parent, "existing data").expect("blocking parent file");
    assert!(
        ScoreStore::write_report(
            &parent.join("report.json"),
            "test",
            "test",
            &TuningProfile::portable(),
            &[],
        )
        .is_err()
    );
    assert_eq!(
        read_to_string(parent).expect("preserved data"),
        "existing data"
    );
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn report_collision_preserves_the_other_writers_temporary_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-report-collision-{}-{nonce}", process_id()));
    create_dir_all(&directory).expect("test directory");
    let path = directory.join("report.json");
    let temporary = path.with_extension("tmp");
    write(&temporary, "other writer").expect("blocking temporary file");
    assert_eq!(
        ScoreStore::write_report(&path, "test", "test", &TuningProfile::portable(), &[])
            .expect_err("exclusive temporary ownership")
            .kind(),
        ErrorKind::AlreadyExists,
    );
    assert_eq!(
        read_to_string(temporary).expect("preserved temporary"),
        "other writer"
    );
    assert!(!path.exists());
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn failed_report_rename_removes_the_owned_temporary_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-report-rename-{}-{nonce}", process_id()));
    let path = directory.join("report.json");
    create_dir_all(&path).expect("blocking destination directory");
    assert!(
        ScoreStore::write_report(&path, "test", "test", &TuningProfile::portable(), &[]).is_err()
    );
    assert!(path.is_dir());
    assert!(!path.with_extension("tmp").exists());
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn complete_source_context_has_a_stable_identity() {
    let first = ScoreStore::measurement_context_hash().expect("readable workspace");
    assert_eq!(ScoreStore::measurement_context_hash(), Some(first));
}

#[test]
fn score_store_save_writes_atomically_without_leaving_temporary_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = temp_dir().join(format!("mp-tune-test-{nonce}"));
    let path = dir.join("test-cache.json");
    let mut store = ScoreStore::load(&path);
    assert!(store.get(42).is_none());

    store.insert(42, vec![100, 200, 300]);
    assert!(path.is_file());
    assert_eq!(read_dir(&dir).expect("directory entries").count(), 1);

    let reloaded = ScoreStore::load(&path);
    assert_eq!(reloaded.get(42), Some(&[100, 200, 300][..]));

    drop(remove_dir_all(dir));
}

#[test]
fn concurrent_cache_saves_leave_a_complete_screening_record() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-concurrent-cache-{}-{nonce}", process_id()));
    let path = directory.join("scores.json");
    let stores: Vec<_> = (1_u64..=8)
        .map(|key| (key, ScoreStore::load(&path)))
        .collect();
    scope(|scope| {
        let handles: Vec<_> = stores
            .into_iter()
            .map(|(key, mut store)| {
                scope.spawn(move || store.insert(key, vec![u128::from(key); 8_192]))
            })
            .collect();
        for handle in handles {
            handle.join().expect("cache writer");
        }
    });
    let source = read_to_string(&path).expect("published cache");
    let entries = ScoreStore::parse_scores(&source).expect("complete cache record");
    assert_eq!(entries.len(), 1);
    for (key, cells) in entries {
        assert!((1..=8).contains(&key));
        assert_eq!(cells, vec![u128::from(key); 8_192]);
    }
    assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn failed_cache_rename_removes_its_temporary_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-blocked-cache-{}-{nonce}", process_id()));
    let path = directory.join("scores.json");
    create_dir_all(&path).expect("blocking destination directory");
    let mut store = ScoreStore::load(&path);
    store.insert(42, vec![10, 20]);
    assert_eq!(store.get(42), Some(&[10, 20][..]));
    assert!(path.is_dir());
    assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    remove_dir_all(directory).expect("test cleanup");
}
