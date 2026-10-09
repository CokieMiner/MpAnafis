//! Portable implementation of `sub_limbs_3_unchecked`.

use super::Limb;

/// Evaluates `dst - borrow * B^len <- src1 - src2` over `len` limbs.
///
/// Subtracts `src2` from `src1` into `dst` and returns the final borrow `b in {0, 1}`.
///
/// # Safety
///
/// - `dst` must cover `len` aligned writable limbs; its contents may be uninitialized.
/// - Both sources must cover `len` aligned initialized readable limbs.
/// - `dst[0..len]` must not overlap `src1[0..len]` or `src2[0..len]`.
/// - `src1` and `src2` may alias each other or be disjoint.
/// - Each byte span must fit in `isize::MAX`; `len == 0` performs no pointer access.
#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the borrow recurrence without a per-operation leaf call"
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
        // SAFETY: i<len addresses initialized aligned limbs in both sources;
        // the read-only sources may alias each other.
        let (diff, b) = unsafe { (*src1.add(i)).borrowing_sub(*src2.add(i), borrow) };
        // SAFETY: i<len addresses an aligned writable destination limb,
        // disjoint from both sources; no old destination value is read.
        unsafe {
            *dst.add(i) = diff;
        }
        borrow = b;
    }
    Limb::from(borrow)
}
