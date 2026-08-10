//! Portable implementation of `add_limbs_3_unchecked`.

use super::Limb;

/// Compute `dst[i] = src1[i] + src2[i] + carry` for `len` limbs, returning
/// the final carry.
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
pub unsafe fn add_limbs_3_unchecked(
    dst: *mut Limb,
    src1: *const Limb,
    src2: *const Limb,
    len: usize,
) -> Limb {
    if len == 1 {
        // SAFETY: The caller guarantees all pointers cover the sole limb.
        let (sum, overflow) = unsafe { (*src1).overflowing_add(*src2) };
        // SAFETY: The caller guarantees dst is writable for the sole limb.
        unsafe {
            *dst = sum;
        }
        return Limb::from(overflow);
    }
    let mut carry = false;
    for i in 0..len {
        // SAFETY: Caller guarantees `src1`, `src2` have at least `len` elements
        let (sum, c) = unsafe { (*src1.add(i)).carrying_add(*src2.add(i), carry) };
        // SAFETY: Caller guarantees `dst` has at least `len` elements
        unsafe {
            *dst.add(i) = sum;
        }
        carry = c;
    }
    Limb::from(carry)
}
