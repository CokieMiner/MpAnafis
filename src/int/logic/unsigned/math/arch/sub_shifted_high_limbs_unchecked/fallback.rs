//! Portable shifted-high subtraction fallback.

use super::Limb;

/// Evaluates `dst_new - borrow_out * B^len = dst_old - K - borrow_in`.
///
/// The cross-limb shifted subtrahend `K` is defined by:
/// `K[i] = (src[i] >> (Limb::BITS - shift)) | (src[i + 1] << shift)` for `i < len - 1`,
/// and `K[len - 1] = src[len - 1] >> (Limb::BITS - shift)`.
///
/// Returns the final subtraction borrow `b in {0, 1}`.
///
/// # Safety
///
/// - `dst` must cover `len` aligned, initialized limbs under exclusive access.
/// - `src` must cover `len` aligned, initialized limbs under shared access.
/// - Spans `dst[0..len]` and `src[0..len]` must not overlap.
/// - Each span's byte length must fit in `isize`; zero length permits null pointers.
/// - `shift` must satisfy `0 < shift < Limb::BITS`.
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
    let mut next_borrow = borrow != 0;
    // SAFETY: the shift contract proves both counts are in 1..Limb::BITS.
    // Every index is below len; the byte bound makes index + 1 representable,
    // and the branch proves that the following source limb exists. The spans
    // are aligned, initialized and disjoint, with exclusive writes to dst.
    unsafe {
        let right_shift = Limb::BITS.unchecked_sub(shift);
        for index in 0..len {
            let low = *src.add(index) >> right_shift;
            let next_index = index.unchecked_add(1);
            let high = if next_index < len {
                *src.add(next_index) << shift
            } else {
                0
            };
            // The fragments occupy disjoint low and high bit ranges.
            let shifted = low | high;
            let (result, underflow) = (*dst.add(index)).borrowing_sub(shifted, next_borrow);
            *dst.add(index) = result;
            next_borrow = underflow;
        }
    }
    Limb::from(next_borrow)
}
