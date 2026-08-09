//! Reference-oracle properties for the selected Montgomery reduction-step kernel.

#![allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "The property calls an unsafe kernel and the oracle indexes equal, non-empty vectors while splitting DoubleLimb products"
)]

use proptest::prelude::*;

use super::{equal_length_odd_limb_vecs, montgomery_inverse};
use crate::int::{
    logic::math::arch::ArchKernels,
    types::{DoubleLimb, Limb},
};

fn reference_montgomery_step(
    out: &mut [Limb],
    multiplier: &[Limb],
    modulus: &[Limb],
    input_limb: Limb,
    inverse: Limb,
) -> Limb {
    let len = multiplier.len();
    if len == 0 {
        return 0;
    }

    let mut input_carry: Limb = 0;
    for index in 0..len {
        let product = (multiplier[index] as DoubleLimb)
            .wrapping_mul(input_limb as DoubleLimb)
            .wrapping_add(out[index] as DoubleLimb)
            .wrapping_add(input_carry as DoubleLimb);
        out[index] = product as Limb;
        input_carry = product.wrapping_shr(Limb::BITS) as Limb;
    }

    let quotient_limb = out[0].wrapping_mul(inverse);
    let mut modulus_carry: Limb = 0;
    for index in 0..len {
        let product = (modulus[index] as DoubleLimb)
            .wrapping_mul(quotient_limb as DoubleLimb)
            .wrapping_add(out[index] as DoubleLimb)
            .wrapping_add(modulus_carry as DoubleLimb);
        out[index] = product as Limb;
        modulus_carry = product.wrapping_shr(Limb::BITS) as Limb;
    }

    assert_eq!(out[0], 0, "the Montgomery inverse must cancel the low limb");
    out.copy_within(1..len, 0);
    let (high_limb, carry_out) = input_carry.overflowing_add(modulus_carry);
    out[len.wrapping_sub(1)] = high_limb;
    Limb::from(carry_out)
}

proptest! {
    #[test]
    fn prop_monty_redc_step_matches_reference(
        case in equal_length_odd_limb_vecs(1..=16),
        input_limb in any::<Limb>(),
    ) {
        let (dst, multiplier, modulus) = case;
        let len = dst.len();
        let inverse = montgomery_inverse(modulus[0]);
        let mut expected_dst = dst.clone();
        let mut actual_dst = dst;

        let expected_carry = reference_montgomery_step(
            &mut expected_dst,
            &multiplier,
            &modulus,
            input_limb,
            inverse,
        );
        // SAFETY: all three vectors contain exactly `len` initialized limbs.
        let actual_carry = unsafe {
            ArchKernels::selected_monty_redc_step_unchecked()(
                actual_dst.as_mut_ptr(),
                multiplier.as_ptr(),
                modulus.as_ptr(),
                len,
                input_limb,
                inverse,
            )
        };

        prop_assert_eq!(
            (actual_carry, &actual_dst),
            (expected_carry, &expected_dst),
            "monty_redc_step mismatch"
        );
    }
}
