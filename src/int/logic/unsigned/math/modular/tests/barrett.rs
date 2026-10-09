//! Barrett quotient bounds, exact residues, and high-product crossovers.

#![expect(
    unsafe_code,
    reason = "Every bounded division case proves modulus <= input < B^(2k) and supplies distinct output and scratch owners."
)]

use super::{BarrettDomain, BarrettScratch, InternalMpUint, LIMB_BITS, Limb, MulScratch};

#[test]
fn reduction_matches_constructed_residues_across_operand_widths() {
    let widths = if cfg!(miri) {
        &[1_usize, 4, 5][..]
    } else {
        &[1_usize, 4, 5, 31, 32, 33, 64][..]
    };
    for &width in widths {
        let bits = width.checked_mul(LIMB_BITS).expect("small test radix");
        let radix = InternalMpUint::one().shl(bits);
        for modulus in [
            radix.sub(&InternalMpUint::one()),
            radix.shr(1).add(&InternalMpUint::one()),
            radix.shr(LIMB_BITS).add(&InternalMpUint::one()),
        ] {
            let domain = BarrettDomain::new(&modulus);
            let mut scratch = BarrettScratch::default();
            let mut multiplication = MulScratch::default();
            let mut output = InternalMpUint::zero();
            for quotient in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                InternalMpUint::from_limb(Limb::MAX),
                InternalMpUint::one().shl(LIMB_BITS),
                radix.clone(),
                radix.square(),
            ] {
                for residue in [
                    InternalMpUint::zero(),
                    InternalMpUint::one(),
                    modulus.sub(&InternalMpUint::one()),
                ] {
                    let input = quotient.mul(&modulus).add(&residue);
                    assert_eq!(input.barrett_reduce(&modulus), residue);
                    domain.reduce_into_with_barrett_scratch(
                        &input,
                        &mut output,
                        &mut multiplication,
                        &mut scratch,
                    );
                    assert_eq!(output, residue);
                }
            }
        }
    }
}

#[test]
fn quotient_and_residue_cross_high_product_boundaries() {
    let widths = if cfg!(miri) {
        &[1_usize, 2, 3][..]
    } else {
        &[
            1_usize, 2, 3, 17, 18, 19, 35, 36, 37, 64, 71, 72, 73, 129, 257,
        ][..]
    };
    for &width in widths {
        let radix = InternalMpUint::one().shl(width * LIMB_BITS);
        for modulus in [
            radix.sub(&InternalMpUint::one()),
            radix.shr(LIMB_BITS),
            radix.shr(LIMB_BITS).add(&InternalMpUint::one()),
        ] {
            let domain = BarrettDomain::new(&modulus);
            let mut scratch = BarrettScratch::default();
            let mut multiplication = MulScratch::default();
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let limit = radix.square();
            let maximum = limit.sub(&InternalMpUint::one());
            for input in [
                InternalMpUint::zero(),
                maximum.clone(),
                modulus.clone(),
                modulus.sub(&InternalMpUint::one()),
                modulus.add(&InternalMpUint::one()),
                modulus.square().sub(&InternalMpUint::one()),
                modulus.square(),
                maximum.shr(LIMB_BITS),
                limit.clone(),
                limit.add(&InternalMpUint::one()),
                maximum,
            ] {
                let expected = input.div_rem(&modulus);
                domain.reduce_into_with_barrett_scratch(
                    &input,
                    &mut remainder,
                    &mut multiplication,
                    &mut scratch,
                );
                assert_eq!(remainder, expected.1);
                if input >= modulus && input < limit {
                    quotient.clone_from(&limit);
                    remainder.clone_from(&limit);
                    // SAFETY: the guard proves modulus <= input < B^(2k).
                    // The domain owns this modulus and all operand, output,
                    // and scratch owners are distinct.
                    unsafe {
                        domain.div_rem_bounded_unchecked(
                            &input,
                            &mut quotient,
                            &mut remainder,
                            &mut multiplication,
                            &mut scratch,
                        );
                    }
                    assert_eq!((&quotient, &remainder), (&expected.0, &expected.1));
                }
            }
        }
    }
}
