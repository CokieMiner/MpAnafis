//! Portable implementation of `add_limbs_unchecked`.

use super::Limb;

/// Add `len` limbs from `src` into `dst` and return the final carry.
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
pub unsafe fn add_limbs_unchecked(dst: *mut Limb, src: *const Limb, len: usize) -> Limb {
    if len == 1 {
        // SAFETY: The caller guarantees both pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*dst).overflowing_add(*src) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    let mut carry = false;
    for i in 0..len {
        // SAFETY: Caller guarantees `dst` and `src` have at least `len` elements
        let (sum, c) = unsafe { (*dst.add(i)).carrying_add(*src.add(i), carry) };
        // SAFETY: Caller guarantees `dst` has at least `len` elements
        unsafe {
            *dst.add(i) = sum;
        }
        carry = c;
    }
    Limb::from(carry)
}
