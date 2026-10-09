//! Profile publication rejects invalid input and preserves complete files.

use std::{
    env::temp_dir,
    fs::{create_dir_all, read_dir, read_to_string, remove_dir_all, write},
    io::ErrorKind,
    process::id as process_id,
    thread::scope,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{ProfileWriter, TuningProfile};

#[test]
fn invalid_profiles_leave_the_destination_and_directory_untouched() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-invalid-profile-{}-{nonce}", process_id()));
    create_dir_all(&directory).expect("test directory");
    let path = directory.join("profile.rs");
    let original = TuningProfile::portable().render("// existing profile");
    write(&path, &original).expect("existing profile");
    let invalid = TuningProfile {
        radix_parse_leaf: 0,
        ..TuningProfile::portable()
    };
    assert!(invalid.validate().is_err());
    for source in [
        "malformed profile".to_owned(),
        invalid.render("// invalid profile"),
    ] {
        assert_eq!(
            ProfileWriter::write_source(&source, &path)
                .expect_err("invalid source")
                .kind(),
            ErrorKind::InvalidInput,
        );
        assert_eq!(read_to_string(&path).expect("existing profile"), original);
        assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    }
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn valid_publication_replaces_the_complete_profile_without_temporary_files() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-valid-profile-{}-{nonce}", process_id()));
    create_dir_all(&directory).expect("test directory");
    let path = directory.join("profile.rs");
    let original = TuningProfile::portable().render("// original profile");
    let replacement = TuningProfile {
        radix_parse_leaf: 8,
        ..TuningProfile::portable()
    };
    write(&path, original).expect("existing profile");
    let source = replacement.render("// replacement profile");
    ProfileWriter::write_source(&source, &path).expect("publish profile");
    let written = read_to_string(&path).expect("published profile");
    assert_eq!(written, source);
    assert_eq!(TuningProfile::from_source(&written), Ok(replacement));
    assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn failed_publication_removes_only_its_own_temporary_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!("mp-tune-blocked-profile-{}-{nonce}", process_id()));
    let path = directory.join("profile.rs");
    create_dir_all(&path).expect("blocking destination directory");
    let preserved = path.join("keep");
    write(&preserved, "existing data").expect("existing data");
    let source = TuningProfile::portable().render("// candidate profile");
    assert!(ProfileWriter::write_source(&source, &path).is_err());
    assert_eq!(
        read_to_string(preserved).expect("preserved data"),
        "existing data"
    );
    assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    remove_dir_all(directory).expect("test cleanup");
}

#[test]
fn concurrent_publications_preserve_one_complete_valid_profile() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = temp_dir().join(format!(
        "mp-tune-concurrent-profile-{}-{nonce}",
        process_id()
    ));
    create_dir_all(&directory).expect("test directory");
    let path = directory.join("profile.rs");
    let sources: Vec<_> = (8..14)
        .map(|leaf| {
            TuningProfile {
                radix_parse_leaf: leaf,
                ..TuningProfile::portable()
            }
            .render("// concurrent profile")
        })
        .collect();
    scope(|scope| {
        let handles: Vec<_> = sources
            .iter()
            .map(|source| {
                let destination = &path;
                scope.spawn(move || ProfileWriter::write_source(source, destination))
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .expect("writer thread")
                .expect("complete publication");
        }
    });
    let written = read_to_string(&path).expect("published profile");
    assert!(sources.contains(&written));
    assert_eq!(
        TuningProfile::from_source(&written)
            .expect("complete source")
            .validate(),
        Ok(())
    );
    assert_eq!(read_dir(&directory).expect("directory entries").count(), 1);
    remove_dir_all(directory).expect("test cleanup");
}
