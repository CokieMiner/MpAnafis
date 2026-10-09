//! Portable fused dual-row multiply-add kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Accumulates two interleaved scalar-product rows and returns their separate carries.
///
/// Empty spans return `(0, 0)` without accessing pointers.
///
/// # Safety
///
/// For nonzero `len`, `src` must cover `len` aligned, initialized limbs and
/// `dst` must cover `len + 1` aligned, initialized, writable limbs. The spans
/// must be disjoint and remain within live allocations of at most `isize::MAX` bytes.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "The casts split a proven two-limb product and sum into low limbs and carries in the generic hot kernel"
)]
#[cfg_attr(
    not(target_pointer_width = "16"),
    expect(
        clippy::cast_possible_truncation,
        reason = "The casts split a proven two-limb product and sum into low limbs and carries in the generic hot kernel"
    )
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    low_scalar: Limb,
    high_scalar: Limb,
) -> (Limb, Limb) {
    let mut low_carry: Limb = 0;
    let mut high_carry: Limb = 0;
    let low_scalar_wide = low_scalar as DoubleLimb;
    let high_scalar_wide = high_scalar as DoubleLimb;

    for i in 0..len {
        // SAFETY: i < len places the aligned read in the initialized source span.
        let source_limb = unsafe { *src.add(i) };
        let source_wide = source_limb as DoubleLimb;

        // SAFETY: i < len and the len+1 destination bound prove i+1 fits and
        // both destination accesses are initialized and writable. The disjoint
        // source cannot be overwritten. Each widened row sum is bounded by
        // (B-1)^2 + 2*(B-1) = B^2-1, so the arithmetic is exact.
        unsafe {
            let next = i.unchecked_add(1);
            let low_carry_wide = low_carry as DoubleLimb;
            let low_destination_wide = (*dst.add(i)) as DoubleLimb;
            let low_sum = source_wide
                .unchecked_mul(low_scalar_wide)
                .unchecked_add(low_carry_wide)
                .unchecked_add(low_destination_wide);

            // Low halves reduce modulo B; shifted high halves fit a Limb.
            // DoubleLimb holds at least 2*LIMB_BITS on 16-, 32-, and 64-bit targets.
            let low_result = low_sum as Limb;
            let next_low_carry = (low_sum >> LIMB_BITS) as Limb;
            *dst.add(i) = low_result;
            low_carry = next_low_carry;

            let high_carry_wide = high_carry as DoubleLimb;
            let high_destination_wide = (*dst.add(next)) as DoubleLimb;
            let high_sum = source_wide
                .unchecked_mul(high_scalar_wide)
                .unchecked_add(high_carry_wide)
                .unchecked_add(high_destination_wide);

            let high_result = high_sum as Limb;
            let next_high_carry = (high_sum >> LIMB_BITS) as Limb;
            *dst.add(next) = high_result;
            high_carry = next_high_carry;
        }
    }
    (low_carry, high_carry)
}
