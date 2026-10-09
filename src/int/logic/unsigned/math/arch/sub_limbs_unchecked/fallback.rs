//! Portable implementation of `sub_limbs_unchecked`.

use super::Limb;

/// Evaluates `dst - borrow * B^len <- dst - src` over `len` limbs.
///
/// Subtracts `len` limbs of `src` from `dst` and returns the final borrow `b in {0, 1}`.
///
/// # Safety
///
/// - `dst` must cover `len` aligned initialized writable limbs.
/// - `src` must cover `len` aligned initialized readable limbs.
/// - Spans `dst[0..len]` and `src[0..len]` must be either completely disjoint
///   or identical pointers (`dst == src`).
/// - Each byte span must fit in `isize::MAX`; `len == 0` performs no pointer access.
#[expect(
    clippy::inline_always,
    reason = "Inlining exposes the borrow recurrence without a per-operation leaf call"
)]
#[inline(always)]
pub unsafe fn sub_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    let mut borrow = false;
    for i in 0..len {
        // SAFETY: i<len addresses initialized aligned limbs in both spans.
        // Both values are read before the destination write, including exact alias.
        let (diff, b) = unsafe { (*dst.add(i)).borrowing_sub(*src.add(i), borrow) };
        // SAFETY: i<len addresses an aligned writable destination limb.
        unsafe {
            *dst.add(i) = diff;
        }
        borrow = b;
    }
    Limb::from(borrow)
}
