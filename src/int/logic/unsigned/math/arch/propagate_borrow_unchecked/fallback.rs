//! Fallback borrow propagation kernel (portable Rust).

use super::Limb;

/// Propagate a borrow through a raw limb pointer slice.
///
/// # Safety
///
/// `dst` must be valid for reading and writing `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
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
