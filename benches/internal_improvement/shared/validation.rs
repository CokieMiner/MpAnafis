//! Untimed correctness checks and one-call warmup for measured kernels.

use mp_anafis::tune_api::Limb;

/// Executes the candidate once in a separate output buffer and verifies it.
pub fn validate_and_warm_product(
    expected: &[Limb],
    label: &str,
    operation: impl FnOnce(&mut [Limb]),
) {
    let mut probe = vec![Limb::MIN; expected.len()];
    operation(&mut probe);
    assert_eq!(probe, expected, "{label} output differs from the reference");
}
