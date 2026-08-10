//! Portable implementation of `sub_limbs_unchecked`.

use super::Limb;

/// Subtract `len` limbs of `src` from `dst` with borrow propagation and
/// return the final borrow-out limb (0 or 1).
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
/// - The `dst` and `src` spans must be identical or disjoint: the kernel
///   reads `src[i]` and `dst[i]` and then writes `dst[i]`, so a partial
///   overlap is a data race.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak performance on platforms without hardware asm"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow = false;
    for i in 0..len {
        // SAFETY: Caller guarantees `dst` and `src` have at least `len` elements
        let (diff, b) = unsafe { (*dst.add(i)).borrowing_sub(*src.add(i), borrow) };
        // SAFETY: Caller guarantees `dst` has at least `len` elements
        unsafe {
            *dst.add(i) = diff;
        }
        borrow = b;
    }
    Limb::from(borrow)
}
