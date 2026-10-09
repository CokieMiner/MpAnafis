//! Buffer sizing and shape validation for raw-limb benchmark kernels.

use super::Limb;

/// Namespace for raw benchmark buffer-contract validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchValidation;

impl BenchValidation {
    /// Validates the destination length for the complete product of `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if the required product limb width overflows `usize` or exceeds `dst.len()`.
    #[must_use]
    #[track_caller]
    pub fn product(dst: &[Limb], a: &[Limb], b: &[Limb]) -> usize {
        let required = a
            .len()
            .checked_add(b.len())
            .expect("benchmark product width overflows usize");
        assert!(
            dst.len() >= required,
            "benchmark destination has {} limbs, but the full product requires {required}",
            dst.len()
        );
        required
    }

    /// Validates the destination length for the complete square of `a`.
    ///
    /// # Panics
    ///
    /// Panics if the required square limb width overflows `usize` or exceeds `dst.len()`.
    #[must_use]
    #[track_caller]
    pub fn square(dst: &[Limb], a: &[Limb]) -> usize {
        let required = a
            .len()
            .checked_mul(2)
            .expect("benchmark square width overflows usize");
        assert!(
            dst.len() >= required,
            "benchmark destination has {} limbs, but the full square requires {required}",
            dst.len()
        );
        required
    }
}
