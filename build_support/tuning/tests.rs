use super::{TuningProfile, profile_for_target};

#[test]
fn every_target_uses_the_single_measured_baseline() {
    let baseline = TuningProfile::portable();
    for (architecture, width) in [
        ("x86_64", "64"),
        ("aarch64", "64"),
        ("powerpc64le", "64"),
        ("riscv64", "64"),
        ("x86", "32"),
        ("wasm32", "32"),
        ("avr", "16"),
    ] {
        assert_eq!(profile_for_target(architecture, width), baseline);
    }
}

#[test]
fn architecture_profiles_preserve_dispatch_order() {
    for (architecture, width) in [
        ("x86_64", "64"),
        ("aarch64", "64"),
        ("powerpc64le", "64"),
        ("riscv64", "64"),
        ("x86", "32"),
        ("wasm32", "32"),
        ("avr", "16"),
    ] {
        let profile = profile_for_target(architecture, width);
        assert!(profile.karatsuba < profile.toom_cook_3);
        assert!(profile.toom_cook_3 < profile.toom_cook_4);
        assert!(profile.toom_cook_4 < profile.toom_cook_6);
        assert!(profile.toom_cook_6 < profile.toom_cook_85);
        assert!(profile.sqr_karatsuba < profile.sqr_toom_cook_3);
        assert!(profile.sqr_toom_cook_3 < profile.sqr_toom_cook_4);
        assert!(profile.sqr_toom_cook_4 < profile.sqr_toom_cook_85);
        assert!(
            profile.sqr_toom_cook_6 == usize::MAX - 1
                || profile.sqr_toom_cook_6 == profile.sqr_toom_cook_85
        );
        assert!(profile.burnikel_ziegler < profile.newton_raphson);
        if profile.ssa != 0 {
            assert!(profile.toom_cook_85 < profile.ssa);
            assert!(profile.sqr_toom_cook_85 < profile.sqr_ssa);
        }
    }
}

#[test]
fn architecture_profiles_pass_semantic_validation() {
    for (architecture, width) in [
        ("x86_64", "64"),
        ("aarch64", "64"),
        ("powerpc64le", "64"),
        ("riscv64", "64"),
        ("x86", "32"),
        ("wasm32", "32"),
        ("avr", "16"),
    ] {
        assert_eq!(profile_for_target(architecture, width).validate(), Ok(()));
    }
}

#[test]
fn baseline_seeds_the_measured_direct_fermat_policy() {
    for (architecture, width) in [
        ("x86_64", "64"),
        ("aarch64", "64"),
        ("powerpc64le", "64"),
        ("riscv64", "64"),
        ("x86", "32"),
        ("wasm32", "32"),
        ("avr", "16"),
    ] {
        let profile = profile_for_target(architecture, width);
        assert_eq!(profile.ssa_direct_fermat_parallel_threshold, 1_048_576);
        assert_eq!(profile.ssa_direct_fermat_parallel_min_workers, 8);
    }
}

#[test]
fn rendered_profiles_round_trip_through_parser() {
    for profile in [
        TuningProfile::portable(),
        profile_for_target("x86_64", "64"),
        profile_for_target("avr", "16"),
    ] {
        let rendered = profile.render("// test profile");
        let parsed_result = TuningProfile::from_source(&rendered);
        assert!(parsed_result.is_ok(), "rendered profile must parse");
        let Ok(parsed_profile) = parsed_result else {
            continue;
        };
        assert_eq!(parsed_profile, profile);
        assert_eq!(parsed_profile.validate(), Ok(()));
    }
}

#[test]
fn semantic_validation_rejects_invalid_disabled_and_sentinel_values() {
    let mut profile = TuningProfile::portable();
    profile.ssa = usize::MAX - 1;
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.burnikel_ziegler_block = 0;
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.toom_cook_4 = profile.toom_cook_3.saturating_sub(1);
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.toom_cook_6 = usize::MAX;
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.balanced_toom8 = profile.karatsuba.saturating_add(1);
    profile.ssa = profile.balanced_toom8;
    assert!(profile.validate().is_err());
}

#[test]
fn semantic_validation_accepts_shadowed_tiers_and_disabled_direct_shift() {
    let mut profile = TuningProfile::portable();
    profile.toom_cook_6 = profile.toom_cook_85;
    profile.sqr_toom_cook_3 = profile.sqr_toom_cook_4;
    profile.ssa_direct_shift_max_limbs = 0;
    assert_eq!(profile.validate(), Ok(()));
}

#[test]
fn semantic_validation_rejects_enabled_tier_after_disabled_tail() {
    let mut profile = TuningProfile::portable();
    profile.toom_cook_6 = usize::MAX - 1;
    assert!(profile.validate().is_err());
}

#[test]
fn semantic_validation_rejects_sentinel_geometry() {
    let mut profile = TuningProfile::portable();
    profile.burnikel_ziegler_block = usize::MAX - 1;
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.ssa_base_modulus_bits = usize::MAX - 1;
    assert!(profile.validate().is_err());
    profile = TuningProfile::portable();
    profile.ssa_direct_shift_max_limbs = usize::MAX - 1;
    assert!(profile.validate().is_err());
}
