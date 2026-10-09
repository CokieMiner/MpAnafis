//! Fallback borrow propagation kernel (portable Rust).

#![expect(
    unsafe_code,
    reason = "The caller provides the initialized writable limb span"
)]

use super::Limb;

/// Absorbs an incoming borrow into `dst[0..len]`, evaluating `dst - borrow_out * B^len <- dst - borrow`.
///
/// Returns the residual borrow `b in {0, 1}`.
///
/// # Safety
///
/// For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
/// within `isize::MAX` bytes. `borrow` must be zero or one.
#[expect(clippy::inline_always, reason = "Inlining retains the first-limb stopping path at its arithmetic caller")]
#[inline(always)]
pub unsafe fn propagate_borrow_unchecked(dst: *mut Limb, len: usize, mut borrow: Limb) -> Limb {
    for i in 0..len {
        // SAFETY: Caller guarantees `dst` is valid for `len` elements, and `i < len`.
        unsafe {
            let (diff, b) = (*dst.add(i)).overflowing_sub(borrow);
            *dst.add(i) = diff;
            borrow = Limb::from(b);
            if borrow == 0 {
                break;
            }
        }
    }
    borrow
}
