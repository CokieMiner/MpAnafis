//! Pure Rust fallback for shift kernels.

use super::{LIMB_BITS, Limb};

/// Left-shift `len` limbs in-place by `shift` bits (`0 < shift < LIMB_BITS`).
/// Returns the bits shifted out of the top limb.
///
/// # Safety
///
/// - `limbs` must be valid for reads and writes of `len` elements.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift` and applies both shift amounts to each element, so
///   an out-of-range amount is undefined behavior.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Critical for peak performance on platforms without hardware asm; \
              LIMB_BITS (≤64 on all targets) always fits in u32"
)]
#[inline(always)]
pub unsafe fn lshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    let mut carry: Limb = 0;
    let c_shift = (LIMB_BITS as u32).wrapping_sub(shift);
    for i in 0..len {
        // SAFETY: Caller guarantees `limbs` has at least `len` elements
        let val = unsafe { *limbs.add(i) };
        // SAFETY: same
        unsafe {
            *limbs.add(i) = val.wrapping_shl(shift) | carry;
        }
        carry = val.wrapping_shr(c_shift);
    }
    carry
}
