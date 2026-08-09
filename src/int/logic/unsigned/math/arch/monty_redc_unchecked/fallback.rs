//! Portable Montgomery reduction step kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Computes one step of Coarsely Integrated Operand Scanning (CIOS) Montgomery reduction.
///
/// For step `i`, this computes:
/// `(out[0..len] + a_i * b[0..len] + q * m[0..len]) / 2^LIMB_BITS`
/// where `q = ((out[0] + a_i * b[0]) * m_inv) mod 2^LIMB_BITS`.
///
/// Stores the shifted result into `out[0..len-1]`, stores the combined low carry into
/// `out[len-1]`, and returns the top overflow carry (either 0 or 1).
///
/// # Safety
/// - `out` must be valid for reads and writes of `len` elements.
/// - `b` and `m` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Critical for peak performance on generic platforms; \
              Limb->DoubleLimb is always a widening cast; dual carry chains guarantee \
              no u128 overflow; low-limb extraction and high-limb carries are exact"
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

    // Step 0: calculate q using element 0
    // SAFETY: caller guarantees len > 0, so index 0 is valid.
    unsafe {
        let out0 = *out;
        let first_multiplicand = *multiplicand;
        let first_modulus = *modulus;

        let first_product = (out0 as DoubleLimb)
            .wrapping_add((first_multiplicand as DoubleLimb).wrapping_mul(scalar_wide));
        let first_low = first_product as Limb;
        let mut product_carry = (first_product >> LIMB_BITS) as Limb;

        let quotient_limb = first_low.wrapping_mul(inverse);
        let quotient_wide = quotient_limb as DoubleLimb;

        let first_reduction = (first_low as DoubleLimb)
            .wrapping_add((first_modulus as DoubleLimb).wrapping_mul(quotient_wide));
        // Low word is guaranteed to be 0 by definition of Montgomery inverse.
        let mut reduction_carry = (first_reduction >> LIMB_BITS) as Limb;

        for j in 1..len {
            let out_j = *out.add(j);
            let multiplicand_limb = *multiplicand.add(j);
            let modulus_limb = *modulus.add(j);

            let product = (out_j as DoubleLimb)
                .wrapping_add((multiplicand_limb as DoubleLimb).wrapping_mul(scalar_wide))
                .wrapping_add(product_carry as DoubleLimb);
            let product_low = product as Limb;
            product_carry = (product >> LIMB_BITS) as Limb;

            let reduction = (product_low as DoubleLimb)
                .wrapping_add((modulus_limb as DoubleLimb).wrapping_mul(quotient_wide))
                .wrapping_add(reduction_carry as DoubleLimb);
            let reduction_low = reduction as Limb;
            reduction_carry = (reduction >> LIMB_BITS) as Limb;

            *out.add(j.wrapping_sub(1)) = reduction_low;
        }

        let (final_sum, final_carry) = product_carry.overflowing_add(reduction_carry);
        *out.add(len.wrapping_sub(1)) = final_sum;
        Limb::from(final_carry)
    }
}
