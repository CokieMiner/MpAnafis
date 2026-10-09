//! Portable Montgomery reduction step kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Evaluates one step of CIOS Montgomery reduction over `len` limbs.
///
/// Given accumulator `out`, multiplicand `multiplicand`, modulus `modulus`,
/// scalar `scalar`, and modular inverse `inverse = -modulus[0]^(-1) mod B`:
///
/// Computes `m = ((out[0] + scalar * multiplicand[0]) * inverse) mod B`, then
/// evaluates `out <- (out + scalar * multiplicand + m * modulus) / B`.
/// Stores the low result into `out[0..len-1]`, the combined carry into `out[len-1]`,
/// and returns the top overflow bit `c in {0, 1}`.
///
/// # Safety
///
/// - For nonzero `len`, all pointers must cover `len` aligned, initialized
///   limbs within `isize::MAX` bytes; `out` must be writable.
/// - `out[0..len]` must not overlap `multiplicand[0..len]` or `modulus[0..len]`.
/// - The modulus must be odd and `inverse * modulus[0] = -1 mod B`.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "Inlining keeps the reduction loop at its caller; Limb-to-DoubleLimb casts widen, and low-word extraction reduces modulo B"
)]
#[cfg_attr(
    not(target_pointer_width = "16"),
    expect(
        clippy::cast_possible_truncation,
        reason = "DoubleLimb narrows to Limb on 32-bit and 64-bit targets"
    )
)]
#[inline(always)]
pub unsafe fn monty_redc_step_unchecked(
    out: *mut Limb,
    multiplicand: *const Limb,
    modulus: *const Limb,
    len: usize,
    scalar: Limb,
    inverse: Limb,
) -> Limb {
    if len == 0 {
        return 0;
    }
    let scalar_wide = scalar as DoubleLimb;

    // SAFETY: len > 0 permits limb zero. Every later index is below len; each
    // shifted store follows the load at j and cannot replace an unread limb.
    // Each product plus two limb-sized addends is at most (B-1)^2+2(B-1)=B^2-1,
    // so widened multiplication and addition are exact. j >= 1 and len > 0
    // make the destination subtractions exact. The inverse cancels limb zero.
    unsafe {
        let out0 = *out;
        let first_multiplicand = *multiplicand;
        let first_modulus = *modulus;

        let first_product = (out0 as DoubleLimb)
            .unchecked_add((first_multiplicand as DoubleLimb).unchecked_mul(scalar_wide));
        let first_low = first_product as Limb;
        let mut product_carry = (first_product >> LIMB_BITS) as Limb;

        let quotient_limb = first_low.wrapping_mul(inverse);
        let quotient_wide = quotient_limb as DoubleLimb;

        let first_reduction = (first_low as DoubleLimb)
            .unchecked_add((first_modulus as DoubleLimb).unchecked_mul(quotient_wide));
        // Low word is guaranteed to be 0 by definition of Montgomery inverse.
        let mut reduction_carry = (first_reduction >> LIMB_BITS) as Limb;

        for j in 1..len {
            let out_j = *out.add(j);
            let multiplicand_limb = *multiplicand.add(j);
            let modulus_limb = *modulus.add(j);

            let product = (out_j as DoubleLimb)
                .unchecked_add((multiplicand_limb as DoubleLimb).unchecked_mul(scalar_wide))
                .unchecked_add(product_carry as DoubleLimb);
            let product_low = product as Limb;
            product_carry = (product >> LIMB_BITS) as Limb;

            let reduction = (product_low as DoubleLimb)
                .unchecked_add((modulus_limb as DoubleLimb).unchecked_mul(quotient_wide))
                .unchecked_add(reduction_carry as DoubleLimb);
            let reduction_low = reduction as Limb;
            reduction_carry = (reduction >> LIMB_BITS) as Limb;

            *out.add(j.unchecked_sub(1)) = reduction_low;
        }

        let (final_sum, final_carry) = product_carry.overflowing_add(reduction_carry);
        *out.add(len.unchecked_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
