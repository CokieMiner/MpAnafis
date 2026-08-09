//! Portable implementation of `sub_limbs_3_unchecked`.

use super::Limb;

/// Compute `dst[i] = src1[i] - src2[i] - borrow` for `len` limbs, returning
/// the final borrow.
///
/// # Safety
///
/// - `dst`, `src1`, and `src2` must each be valid for `len` elements.
/// - `dst` must not overlap either input span: the kernel writes `dst`
///   while it reads `src1` and `src2`.
/// - `src1` and `src2` are read-only and may alias each other.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak performance on platforms without hardware asm"
)]
#[inline(always)]
pub unsafe fn sub_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    let mut borrow = false;
    for i in 0..len {
        // SAFETY: Caller guarantees `src1`, `src2` have at least `len` elements
        let (diff, b) = unsafe { (*src1.add(i)).borrowing_sub(*src2.add(i), borrow) };
        // SAFETY: Caller guarantees `dst` has at least `len` elements
        unsafe {
            *dst.add(i) = diff;
        }
        borrow = b;
    }
    Limb::from(borrow)
}
