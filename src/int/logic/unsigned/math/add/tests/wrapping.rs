//! Negative residues with normalized-away high zero limbs.

use super::super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
fn subtraction_restores_residue_width_before_sign_extension() {
    // For b = B^n - B, the residue of -b modulo B^n is B. A wider negative
    // residue restores the discarded zero columns before appending ones.
    for width in [2, 3, 4, 5, 8, 16] {
        let mut words = alloc::vec![Limb::MAX; width];
        *words.first_mut().expect("nonempty operand") = 0;
        let b = InternalMpUint::from_limbs(words);
        for extra in [0, 1, LIMB_BITS - 1, LIMB_BITS, LIMB_BITS + 1] {
            let bits = width * LIMB_BITS + extra;
            let mut expected = alloc::vec![0; width];
            *expected.get_mut(1).expect("at least two limbs") = 1;
            expected.resize(bits.div_ceil(LIMB_BITS), Limb::MAX);
            if bits.div_ceil(LIMB_BITS) > width && !bits.is_multiple_of(LIMB_BITS) {
                *expected.last_mut().expect("nonempty residue") &=
                    Limb::MAX >> (LIMB_BITS - bits % LIMB_BITS);
            }
            assert_eq!(
                InternalMpUint::zero().wrapping_sub_with_underflow(&b, bits),
                (InternalMpUint::from_limbs(expected), true)
            );
            assert_eq!(
                b.wrapping_sub_with_underflow(&b, bits),
                (InternalMpUint::zero(), false)
            );
            assert_eq!(
                b.wrapping_sub_with_underflow(&InternalMpUint::zero(), bits),
                (b.clone(), false)
            );
        }
    }
}
