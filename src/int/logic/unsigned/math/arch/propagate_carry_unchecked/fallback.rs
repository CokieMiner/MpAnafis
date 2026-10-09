//! Fallback carry propagation kernel (portable Rust).

#![expect(
    unsafe_code,
    reason = "The caller provides the initialized writable limb span"
)]

use super::Limb;

/// Absorbs an incoming carry into `dst[0..len]`, evaluating `dst + carry_out * B^len <- dst + carry`.
///
/// Returns the residual carry `c in {0, 1}`.
///
/// # Safety
///
/// For nonzero `len`, `dst` covers `len` aligned, initialized, writable limbs
/// within `isize::MAX` bytes. `carry` must be zero or one.
#[expect(clippy::inline_always, reason = "Inlining retains the first-limb stopping path at its arithmetic caller")]
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
