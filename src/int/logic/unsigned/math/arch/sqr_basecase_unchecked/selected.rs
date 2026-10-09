//! Compile-time square selection for targets with BMI2.

#![expect(
    unsafe_code,
    reason = "The caller provides disjoint input/output spans; module cfg proves CPU features"
)]

use super::{Limb, large_square, small_square};

/// Writes the complete `2 * len` limbs of `source^2`.
///
/// # Safety
///
/// `source` must cover `len` aligned initialized readable limbs. `dst` must
/// cover `2 * len` aligned writable limbs, disjoint from `source`; its contents
/// may be uninitialized. The complete output byte span must fit in `isize::MAX`.
/// `len == 0` performs no pointer access. The parent cfg establishes BMI2 and,
/// when selecting the ADX backend, ADX support.
#[inline]
pub unsafe fn sqr_basecase_unchecked(dst: *mut Limb, source: *const Limb, len: usize) {
    if len <= 8 {
        // SAFETY: the caller provides complete disjoint spans. The portable
        // small-width implementation initializes the complete square.
        unsafe {
            small_square(dst, source, len);
        }
    } else {
        // SAFETY: the same complete spans satisfy the large-width contract.
        // The parent cfg proves BMI2 and, for the ADX provider, ADX support.
        unsafe {
            large_square(dst, source, len);
        }
    }
}
