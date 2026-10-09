//! Tests for platform cache and CPU-list parsing without host access.

use super::Platform;

#[test]
fn parses_linux_cache_sizes_without_host_access() {
    assert_eq!(Platform::parse_size_bytes("32K"), Some(32 * 1024));
    assert_eq!(Platform::parse_size_bytes(" 2M\n"), Some(2 * 1024 * 1024));
    assert_eq!(Platform::parse_size_bytes("4096"), Some(4096));
    assert_eq!(Platform::parse_size_bytes("3.5M"), None);
    assert_eq!(Platform::parse_size_bytes("bad"), None);
}

#[test]
fn parses_partial_cache_metadata() {
    let cache = Platform::parse_cache_identity(
        " 3\n",
        Some(" Unified\n"),
        Some("16M\n"),
        Some("64\n"),
        Some("0-3,8\n"),
    );
    assert_eq!(cache.level, Some(3));
    assert_eq!(cache.kind.as_deref(), Some("Unified"));
    assert_eq!(cache.size_bytes, Some(16 * 1024 * 1024));
    assert_eq!(cache.coherency_line_bytes, Some(64));
    assert_eq!(cache.shared_cpu_list.as_deref(), Some("0-3,8"));

    let partial = Platform::parse_cache_identity("unknown", Some("\n"), Some("bad"), None, None);
    assert_eq!(partial.level, None);
    assert_eq!(partial.kind, None);
    assert_eq!(partial.size_bytes, None);
}

#[test]
fn cache_ordering_and_key_are_deterministic() {
    let high =
        Platform::parse_cache_identity("3", Some("Unified"), Some("32M"), Some("64"), Some("0-7"));
    let low = Platform::parse_cache_identity("1", Some("Data"), Some("32K"), Some("64"), Some("0"));
    assert!(Platform::cache_ordering(&low, &high).is_lt());
    assert_eq!(
        Platform::cache_key(&low),
        "l=1;t=Data;s=32768;line=64;shared=0"
    );
}

#[test]
fn parses_linux_cpu_lists_without_host_access() {
    assert!(Platform::cpu_list_contains("0-3,8,10-12", 0));
    assert!(Platform::cpu_list_contains("0-3,8,10-12", 11));
    assert!(!Platform::cpu_list_contains("0-3,8,10-12", 9));
    assert!(!Platform::cpu_list_contains("bad,8-x", 8));
}

#[test]
fn explicit_parallel_cpu_sets_are_unique_and_ordered() {
    assert_eq!(Platform::parse_cpu_list("6,2-4"), Ok(vec![2, 3, 4, 6]));
    for invalid in ["", "4-2", "1,1", "1-3,2", "bad", "1,"] {
        assert!(
            Platform::parse_cpu_list(invalid).is_err(),
            "invalid CPU set {invalid}"
        );
    }
    assert!(Platform::parse_cpu_list(&format!("0-{}", usize::MAX)).is_err());
}

#[test]
fn host_hygiene_probes_never_fail_without_sysfs() {
    // Both probes read live host state, so the test only requires totality:
    // a detected L2 is a positive size and any warning is non-empty.
    if let Some(l2) = Platform::l2_cache_bytes() {
        assert!(l2 > 0);
    }
    if let Some(warning) = Platform::frequency_stability_warning() {
        assert_ne!(warning, "");
    }
}
