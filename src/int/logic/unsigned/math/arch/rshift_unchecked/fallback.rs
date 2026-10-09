//! Pure Rust fallback for shift kernels.

use super::{LIMB_BITS, Limb};

/// Evaluates `limbs <- limbs >> shift` in place over `len` limbs.
///
/// Returns the bits shifted out of the least significant limb:
/// `limbs[0] << (LIMB_BITS - shift)`.
///
/// # Safety
///
/// - For nonzero `len`, `limbs` covers `len` aligned, initialized, writable
///   limbs within `isize::MAX` bytes.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Inlining keeps the limb loop at its caller; LIMB_BITS is 16, 32, or 64 and fits in u32"
)]
#[inline(always)]
pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    let mut carry: Limb = 0;
    // SAFETY: 0 < shift < LIMB_BITS makes the complementary count exact.
    let c_shift = unsafe { (LIMB_BITS as u32).unchecked_sub(shift) };
    for i in (0..len).rev() {
        // SAFETY: i < len selects an aligned initialized limb in the span.
        let val = unsafe { *limbs.add(i) };
        // SAFETY: the same limb is writable; its original value was loaded
        // before the store and carry contains only the higher limb's bits.
        unsafe {
            *limbs.add(i) = (val >> shift) | carry;
        }
        carry = val << c_shift;
    }
    carry
}
