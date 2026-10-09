//! Target selection and default-profile validity.

use super::super::TuningProfile;

#[test]
fn target_selection_preserves_the_valid_portable_profile() {
    let baseline = TuningProfile::portable();
    assert_eq!(baseline.validate(), Ok(()), "built-in profile is valid");
    assert_eq!(TuningProfile::default(), baseline, "default is portable");
    for architecture in [
        "x86_64",
        "aarch64",
        "powerpc64le",
        "riscv64",
        "x86",
        "wasm32",
        "avr",
        "unknown",
        "",
    ] {
        for width in ["16", "32", "64", "unknown", ""] {
            assert_eq!(
                TuningProfile::for_target(architecture, width),
                baseline,
                "target {architecture:?}, pointer width {width:?}"
            );
        }
    }
}
