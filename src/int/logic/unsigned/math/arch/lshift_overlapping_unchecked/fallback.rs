//! Portable overlap-safe left shift.

use super::Limb;

/// Shift `limbs[..len]` into `limbs[offset..offset + len]`, where `offset` may
/// be zero.
///
/// # Safety
///
/// `limbs` must cover `offset + len` initialized writable limbs, and `shift`
/// must be in `1..Limb::BITS`.
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
    let drop = Limb::BITS.wrapping_sub(shift);
    // SAFETY: len is nonzero and the caller provides the complete source.
    let carry = unsafe { *limbs.add(len.wrapping_sub(1)) >> drop };
    let mut index = len.wrapping_sub(1);
    while index != 0 {
        // SAFETY: both source indices are below len and the destination index
        // is below offset + len. Descending stores cannot touch a lower unread
        // source limb because offset + index >= index.
        unsafe {
            *limbs.add(offset.wrapping_add(index)) =
                (*limbs.add(index) << shift) | (*limbs.add(index.wrapping_sub(1)) >> drop);
        }
        index = index.wrapping_sub(1);
    }
    // SAFETY: source zero and destination offset are in the caller's span.
    unsafe {
        *limbs.add(offset) = *limbs << shift;
    }
    carry
}
