//! Portable overlap-safe left shift.

use super::Limb;

/// Evaluates `limbs[offset..offset + len] <- (limbs[0..len] << shift) mod B^len`.
///
/// Shifts a limb prefix into an overlapping higher position using descending
/// traversal, returning the bits shifted out of the most significant limb:
/// `limbs[len - 1] >> (Limb::BITS - shift)`.
///
/// # Safety
///
/// - For nonzero `len`, `limbs` must cover `offset + len` aligned, initialized,
///   writable limbs within `isize::MAX` bytes.
/// - `offset + len` must not overflow `usize`.
/// - `shift` must satisfy `0 < shift < Limb::BITS`.
#[inline]
pub unsafe fn lshift_overlapping_unchecked(
    limbs: *mut Limb,
    len: usize,
    offset: usize,
    shift: u32,
) -> Limb {
    if len == 0 {
        return 0;
    }
    // SAFETY: len > 0 and 0 < shift < Limb::BITS make both subtractions exact.
    // index < len and the caller's offset+len bound prove offset+index fits.
    // Descending traversal loads each source before any overlapping higher
    // store can replace it. All accesses stay within the aligned writable span.
    unsafe {
        let drop = Limb::BITS.unchecked_sub(shift);
        let carry = *limbs.add(len).sub(1) >> drop;
        for index in (1..len).rev() {
            *limbs.add(offset.unchecked_add(index)) =
                (*limbs.add(index) << shift) | (*limbs.add(index).sub(1) >> drop);
        }
        *limbs.add(offset) = *limbs << shift;
        carry
    }
}
