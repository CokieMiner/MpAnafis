//! CPU-feature policy and process-stable selection.

#[cfg(debug_assertions)]
use std::env::var;
use std::{arch::is_x86_feature_detected, thread};

#[cfg(not(debug_assertions))]
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
use super::super::{X86Backend, selected_x86_backend};
use super::super::{X86SimdTier, selected_x86_simd_tier};

#[test]
fn feature_policies_cover_all_combinations_and_reject_unsupported_requests() {
    for first in [false, true] {
        for second in [false, true] {
            let expected_simd = if first && second {
                X86SimdTier::Avx512
            } else if first {
                X86SimdTier::Avx2
            } else {
                X86SimdTier::Sse2
            };
            assert_eq!(
                X86SimdTier::from_features(first, second, None),
                expected_simd
            );
            for request in [
                "avx512", "avx2", "sse2", "adx", "bmi2", "vanilla", "", "unknown",
            ] {
                let expected = if request == "avx512" && first && second {
                    X86SimdTier::Avx512
                } else if request == "avx2" && first {
                    X86SimdTier::Avx2
                } else {
                    X86SimdTier::Sse2
                };
                assert_eq!(
                    X86SimdTier::from_features(first, second, Some(request)),
                    expected
                );
            }
            #[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
            {
                let expected_arithmetic = match (first, second) {
                    (true, true) => X86Backend::AdxBmi2,
                    (true, false) => X86Backend::Adx,
                    (false, true) => X86Backend::Bmi2,
                    (false, false) => X86Backend::Baseline,
                };
                assert_eq!(
                    X86Backend::from_features(first, second, None),
                    expected_arithmetic
                );
                for request in [
                    "adx", "bmi2", "vanilla", "avx512", "avx2", "sse2", "", "unknown",
                ] {
                    let expected = match request {
                        "adx" if first && second => X86Backend::AdxBmi2,
                        "adx" if first => X86Backend::Adx,
                        "bmi2" if second => X86Backend::Bmi2,
                        _ => X86Backend::Baseline,
                    };
                    assert_eq!(
                        X86Backend::from_features(first, second, Some(request)),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn cached_selection_matches_host_features_and_is_shared_across_threads() {
    #[cfg(debug_assertions)]
    let requested = var("MP_ANAFIS_TEST_BACKEND").ok();
    #[cfg(not(debug_assertions))]
    let requested: Option<String> = None;
    let expected_simd = X86SimdTier::from_features(
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f"),
        requested.as_deref(),
    );
    #[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
    let expected_arithmetic = X86Backend::from_features(
        is_x86_feature_detected!("adx"),
        is_x86_feature_detected!("bmi2"),
        requested.as_deref(),
    );
    let threads = (0..4)
        .map(|_| {
            thread::spawn(move || {
                for _ in 0..16 {
                    assert_eq!(selected_x86_simd_tier(), expected_simd);
                    #[cfg(not(all(target_feature = "adx", target_feature = "bmi2")))]
                    assert_eq!(selected_x86_backend(), expected_arithmetic);
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in threads {
        worker.join().expect("selection worker");
    }
}
