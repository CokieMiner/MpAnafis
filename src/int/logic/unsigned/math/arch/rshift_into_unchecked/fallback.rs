//! Pure Rust fallback for the out-of-place right-shift kernel.

use super::{LIMB_BITS, Limb};

/// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[0] << (64-shift)`, the
/// bits shifted out of the bottom limb.
///
/// # Safety
///
/// - `dst` must be valid for writes of `len` elements and `src` for reads of
///   `len` elements.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift`, so an out-of-range amount is undefined behavior.
/// - `dst` and `src` must not overlap, even partially: the kernel reads
///   `src` while it writes `dst`.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Critical for peak performance on platforms without hardware asm; \
              LIMB_BITS (≤64 on all targets) always fits in u32"
)]
#[inline(always)]
pub unsafe fn rshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    let c_shift = (LIMB_BITS as u32).wrapping_sub(shift);
    let mut carry: Limb = 0;
    for i in (0..len).rev() {
        // SAFETY: Caller guarantees `dst` writable and `src` readable for
        // `len` elements, shift in 1..LIMB_BITS, and no aliasing.
        let val = unsafe { *src.add(i) };
        // SAFETY: same span guarantee
        unsafe {
            *dst.add(i) = val.wrapping_shr(shift) | carry;
        }
        carry = val.wrapping_shl(c_shift);
    }
    carry
}
