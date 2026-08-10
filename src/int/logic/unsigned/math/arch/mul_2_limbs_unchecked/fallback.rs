//! Portable write-only dual-row multiplication kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Write `src * (s0 + s1 * B)` into `dst` without reading its old contents.
///
/// The overlapping rows are staged so that the row-one limb at `dst[i]` is
/// consumed when row zero reaches the same position.  The final two limbs are
/// then closed with the two row carries.
///
/// # Safety
///
/// `src` must be valid for `len` limbs and `dst` for `len + 2` limbs.  The
/// caller must not alias either input with the destination.  `len == 0`
/// returns without dereferencing either pointer.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The casts split a proven two-limb product into its low limb and carry in the generic hot kernel"
)]
#[inline(always)]
pub unsafe fn mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    s0: Limb,
    s1: Limb,
) {
    if len == 0 {
        return;
    }

    let scalar0 = s0 as DoubleLimb;
    let scalar1 = s1 as DoubleLimb;

    // SAFETY: len > 0 and the caller guarantees the source has len limbs.
    let first = unsafe { *src } as DoubleLimb;
    let product0 = first.wrapping_mul(scalar0);
    let product1 = first.wrapping_mul(scalar1);

    // SAFETY: the caller guarantees dst has len+2 writable limbs.
    unsafe {
        *dst = product0 as Limb;
        *dst.add(1) = product1 as Limb;
    }
    let mut carry0 = (product0 >> LIMB_BITS) as Limb;
    let mut carry1 = (product1 >> LIMB_BITS) as Limb;

    let mut index = 1_usize;
    while index < len {
        // SAFETY: index < len, so the source read is within src[0..len].
        let value = unsafe { *src.add(index) } as DoubleLimb;
        let loop_prod0 = value.wrapping_mul(scalar0);
        let loop_prod1 = value.wrapping_mul(scalar1);

        // dst[index] already contains row one from the preceding iteration;
        // row one at dst[index+1] is still unwritten.
        // SAFETY: both destination positions are within dst[0..len+2].
        unsafe {
            let row0 = loop_prod0
                .wrapping_add(carry0 as DoubleLimb)
                .wrapping_add(*dst.add(index) as DoubleLimb);
            *dst.add(index) = row0 as Limb;
            carry0 = (row0 >> LIMB_BITS) as Limb;

            let row1 = loop_prod1.wrapping_add(carry1 as DoubleLimb);
            *dst.add(index.wrapping_add(1)) = row1 as Limb;
            carry1 = (row1 >> LIMB_BITS) as Limb;
        }
        index = index.wrapping_add(1);
    }

    // The final row-zero carry overlaps the final row-one limb.
    // SAFETY: len and len+1 are within dst[0..len+2].
    unsafe {
        let final_sum = (*dst.add(len) as DoubleLimb).wrapping_add(carry0 as DoubleLimb);
        *dst.add(len) = final_sum as Limb;
        let (top, overflow) = carry1.overflowing_add((final_sum >> LIMB_BITS) as Limb);
        debug_assert!(!overflow, "two-row product exceeded len+2 limbs");
        *dst.add(len.wrapping_add(1)) = top;
    }
}
