//! Fallback carry propagation kernel (portable Rust).

use super::Limb;

/// Propagate a carry through a raw limb pointer slice.
///
/// # Safety
///
/// `dst` must be valid for reading and writing `len` elements of type `Limb`.
#[allow(clippy::inline_always, reason = "Critical for peak performance")]
#[inline(always)]
pub unsafe fn propagate_carry_unchecked(dst: *mut Limb, len: usize, mut carry: Limb) -> Limb {
    for i in 0..len {
        // SAFETY: Caller guarantees `dst` is valid for `len` elements, and `i < len`.
        unsafe {
            let (sum, c) = (*dst.add(i)).overflowing_add(carry);
            *dst.add(i) = sum;
            carry = Limb::from(c);
            if carry == 0 {
                break;
            }
        }
    }
    carry
}
