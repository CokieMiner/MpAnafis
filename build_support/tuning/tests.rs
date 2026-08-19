use super::{TuningProfile, profile_for_target};

#[test]
fn pointer_width_precedes_isa_specific_profile() {
    let x32 = profile_for_target("x86_64", "32");
    let aarch64_ilp32 = profile_for_target("aarch64", "32");
    let hypothetical_x86_16 = profile_for_target("x86_64", "16");

    assert_eq!(x32.karatsuba, 28);
    assert_eq!(x32.ssa, 8_192);
    assert_eq!(aarch64_ilp32.karatsuba, 28);
    assert_eq!(aarch64_ilp32.ssa, 8_192);
    assert_eq!(hypothetical_x86_16.karatsuba, 32);
    assert_eq!(hypothetical_x86_16.ssa, 0);
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
fn rendered_profiles_round_trip_through_parser() {
    for profile in [
        TuningProfile::portable(),
        profile_for_target("x86_64", "64"),
        profile_for_target("avr", "16"),
    ] {
        let rendered = profile.render("// test profile");
        let parsed = TuningProfile::from_source(&rendered).expect("rendered profile parses");
        assert_eq!(parsed, profile);
        assert_eq!(parsed.validate(), Ok(()));
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
