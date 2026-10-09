//! Portable write-only dual-row multiplication kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Evaluates `dst[0..len + 2] <- src * s0 + (src * s1) * B` over `len` limbs.
///
/// Computes two scalar-product rows and writes the combined product into `dst`
/// without reading its prior contents.
///
/// # Safety
///
/// - For nonzero `len`, aligned `dst` must cover `len + 2` writable limbs,
///   which may be uninitialized; aligned `src` must cover `len` initialized limbs.
/// - `len + 2` must fit `usize`, and both byte spans must fit `isize::MAX`.
/// - `dst[0..len + 2]` and `src[0..len]` must not overlap.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "The casts split a proven two-limb product into its low limb and carry in the generic hot kernel"
)]
#[cfg_attr(
    not(target_pointer_width = "16"),
    expect(
        clippy::cast_possible_truncation,
        reason = "DoubleLimb narrows to Limb on 32-bit and 64-bit targets"
    )
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
    // SAFETY: widened limb products are at most (B-1)^2 < B^2.
    let (product0, product1) = unsafe { (first.unchecked_mul(scalar0), first.unchecked_mul(scalar1)) };

    // SAFETY: the caller guarantees dst has len+2 writable limbs.
    unsafe {
        *dst = product0 as Limb;
        *dst.add(1) = product1 as Limb;
    }
    let mut carry0 = (product0 >> LIMB_BITS) as Limb;
    let mut carry1 = (product1 >> LIMB_BITS) as Limb;

    for index in 1..len {
        // SAFETY: index < len, so the source read is within src[0..len].
        let value = unsafe { *src.add(index) } as DoubleLimb;

        // dst[index] already contains row one from the preceding iteration;
        // row one at dst[index+1] is still unwritten.
        // SAFETY: index < len and the len+2 span bound make index+1 exact.
        // dst[index] was initialized by the preceding row-one store. Each
        // product plus two limbs is <= (B-1)^2+2(B-1)=B^2-1, so all widened
        // operations are exact on 16-, 32-, and 64-bit limb targets.
        unsafe {
            let row0 = value.unchecked_mul(scalar0)
                .unchecked_add(carry0 as DoubleLimb)
                .unchecked_add(*dst.add(index) as DoubleLimb);
            *dst.add(index) = row0 as Limb;
            carry0 = (row0 >> LIMB_BITS) as Limb;

            let row1 = value.unchecked_mul(scalar1).unchecked_add(carry1 as DoubleLimb);
            *dst.add(index.unchecked_add(1)) = row1 as Limb;
            carry1 = (row1 >> LIMB_BITS) as Limb;
        }
    }

    // The final row-zero carry overlaps the final row-one limb.
    // SAFETY: len and len+1 lie in the writable len+2 span; dst[len] was
    // initialized by the last row-one store. Its sum with carry0 is < 2B.
    // For s1 > 0, row one's high carry is <= s1-1; adding the final bit is
    // <= s1. For s1=0 both that carry and the pending limb are zero.
    unsafe {
        let final_sum = (*dst.add(len) as DoubleLimb).unchecked_add(carry0 as DoubleLimb);
        *dst.add(len) = final_sum as Limb;
        let top = carry1.unchecked_add((final_sum >> LIMB_BITS) as Limb);
        *dst.add(len.unchecked_add(1)) = top;
    }
}
