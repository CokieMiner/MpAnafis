//! Portable shifted-high subtraction fallback.

use super::Limb;

/// Subtract a cross-limb shifted source span from `dst`, including `borrow`.
///
/// For every `i < len`, the subtrahend limb is
/// `(src[i] >> (Limb::BITS - shift)) | (src[i + 1] << shift)`, with the
/// out-of-range `src[len]` term defined as zero.
///
/// # Safety
///
/// - `dst` and `src` must each cover `len` limbs.
/// - Their active spans must not overlap.
/// - `0 < shift < Limb::BITS`.
/// - `borrow <= 1`.
pub unsafe fn sub_shifted_high_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
    borrow: Limb,
) -> Limb {
    debug_assert!(
        shift > 0 && shift < Limb::BITS,
        "the cross-limb shift must be strictly inside one limb"
    );
    debug_assert!(borrow <= 1, "a subtraction borrow is one bit");
    let right_shift = Limb::BITS.wrapping_sub(shift);
    let mut next_borrow = borrow != 0;
    let mut index = 0_usize;
    while index < len {
        // SAFETY: index < len and the caller provides the complete source span.
        let low = unsafe { *src.add(index) }.wrapping_shr(right_shift);
        let high = if index.wrapping_add(1) < len {
            // SAFETY: the branch proves index + 1 < len.
            unsafe { *src.add(index.wrapping_add(1)) }.wrapping_shl(shift)
        } else {
            0
        };
        // The shifted fragments occupy disjoint bit ranges, so addition and OR
        // are identical. Keep the source expression explicit in the fallback.
        let shifted = low | high;
        // SAFETY: index < len and the caller provides the complete destination span.
        let minuend = unsafe { *dst.add(index) };
        let (partial, underflow_a) = minuend.overflowing_sub(shifted);
        let (result, underflow_b) = partial.overflowing_sub(Limb::from(next_borrow));
        // SAFETY: index < len.
        unsafe {
            *dst.add(index) = result;
        }
        next_borrow = underflow_a | underflow_b;
        index = index.wrapping_add(1);
    }
    Limb::from(next_borrow)
}
