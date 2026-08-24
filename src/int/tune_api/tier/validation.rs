//! Shared benchmark buffer validation.

use super::Limb;

/// Namespace for raw benchmark buffer-contract validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchValidation;

impl BenchValidation {
    #[track_caller]
    pub fn product(dst: &[Limb], a: &[Limb], b: &[Limb]) {
        assert!(
            a.len() <= usize::MAX.wrapping_sub(b.len()),
            "benchmark product width overflows usize"
        );
        // The preceding boundary check proves this sum cannot wrap.
        let required = a.len().wrapping_add(b.len());
        assert!(
            dst.len() >= required,
            "benchmark destination has {} limbs, but the full product requires {required}",
            dst.len()
        );
    }

    #[track_caller]
    pub fn square(dst: &[Limb], a: &[Limb]) {
        assert!(
            a.len() <= usize::MAX.wrapping_shr(1),
            "benchmark square width overflows usize"
        );
        // The preceding boundary check proves this product cannot wrap.
        let required = a.len().wrapping_mul(2);
        assert!(
            dst.len() >= required,
            "benchmark destination has {} limbs, but the full square requires {required}",
            dst.len()
        );
    }

    #[track_caller]
    pub fn scratch(scratch: &[Limb], required: usize) {
        assert!(
            scratch.len() >= required,
            "benchmark scratch has {} limbs, but this tier requires {required}",
            scratch.len()
        );
    }
}
