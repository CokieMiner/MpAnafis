//! Cyclic residue folding, tail widths, and the two zero encodings.

use crate::int::logic::unsigned::math::div::newton::products::newton_wrapped_remainder;

use super::{DivScratch, InternalMpUint, Limb};

#[test]
fn cyclic_residue_width_depends_on_the_reconstructed_wrap() {
    for n in [2_usize, 4] {
        let mut denominator = alloc::vec![0; n];
        *denominator.first_mut().expect("low divisor limb") = 1;
        *denominator.last_mut().expect("high divisor limb") = 1 << (Limb::BITS - 1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let estimate = InternalMpUint::from_limbs(alloc::vec![7, 1]);
        let product = divisor.mul(&estimate);
        let modulus = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n]);
        let cyclic_product = product.rem(&modulus);
        let mut scratch = DivScratch::default();
        scratch.q_den_low.extend_from_slice(cyclic_product.limbs());
        scratch.q_den_low.resize(n, 0);
        for expected in [
            InternalMpUint::from_limb(3),
            divisor.shl(1).add(&InternalMpUint::from_limb(3)),
        ] {
            let mut input = product.add(&expected).limbs().to_vec();
            input.resize(n.checked_mul(2).expect("bounded cyclic dividend"), 0);
            let residue_mod8 = expected.limbs().first().copied().expect("nonzero residue") & 7;
            let (fold, tail) = input.split_at_mut(n);
            let width =
                newton_wrapped_remainder(fold, tail, residue_mod8, scratch.q_den_low.as_slice());
            let residue = input.get(..width).expect("initialized cyclic residue");
            assert_eq!(InternalMpUint::from_limbs_slice(residue), expected);
            let (_, higher) = residue.split_at(n);
            assert_eq!(higher.len(), usize::from(expected.limbs().len() > n));
            assert_eq!(higher.first().copied().unwrap_or(0), Limb::from(width > n));
        }
    }
}

#[test]
fn cyclic_folding_accepts_a_single_limb_tail() {
    for n in [2_usize, 3, 4, 17] {
        let mut denominator = alloc::vec![0; n];
        *denominator.first_mut().expect("low divisor limb") = 1;
        *denominator.last_mut().expect("high divisor limb") = 1 << (Limb::BITS - 1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let estimate = InternalMpUint::from_limbs(alloc::vec![7, 1]);
        let product = divisor.mul(&estimate);
        let width = n.checked_add(1).expect("cyclic width");
        let modulus = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let cyclic_product = product.rem(&modulus);
        let mut scratch = DivScratch::default();
        scratch.q_den_low.extend_from_slice(cyclic_product.limbs());
        scratch.q_den_low.resize(width, 0);
        for error in 0..=3 {
            for remainder in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                let expected = divisor
                    .mul(&InternalMpUint::from_limb(error))
                    .add(&remainder);
                let mut input = product.add(&expected).limbs().to_vec();
                input.resize(width.checked_add(1).expect("single tail limb"), 0);
                let residue_mod8 = expected.limbs().first().copied().unwrap_or(0) & 7;
                let (fold, tail) = input.split_at_mut(width);
                let active = newton_wrapped_remainder(
                    fold,
                    tail,
                    residue_mod8,
                    scratch.q_den_low.as_slice(),
                );
                assert_eq!(
                    InternalMpUint::from_limbs_slice(input.get(..active).expect("residue")),
                    expected
                );
            }
        }
    }
}

#[test]
fn cyclic_zero_and_modulus_multiples_have_distinct_wrap_counts() {
    for n in [2_usize, 4] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n]);
        let estimate = InternalMpUint::from_limbs(alloc::vec![7, 1]);
        let product = divisor.mul(&estimate);
        let mut scratch = DivScratch::default();
        for zero_encoding in [0, Limb::MAX] {
            scratch.q_den_low.clear();
            scratch.q_den_low.resize(n, zero_encoding);
            for error in 0..=3 {
                for remainder in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                    let expected = divisor
                        .mul(&InternalMpUint::from_limb(error))
                        .add(&remainder);
                    let mut input = product.add(&expected).limbs().to_vec();
                    input.resize(n.checked_mul(2).expect("bounded cyclic dividend"), 0);
                    let residue_mod8 = expected.limbs().first().copied().unwrap_or(0) & 7;
                    let (fold, tail) = input.split_at_mut(n);
                    let width = newton_wrapped_remainder(
                        fold,
                        tail,
                        residue_mod8,
                        scratch.q_den_low.as_slice(),
                    );
                    let residue = input.get(..width).expect("initialized cyclic residue");
                    assert_eq!(InternalMpUint::from_limbs_slice(residue), expected);
                    assert_eq!(
                        width == 0,
                        error == 0 && remainder.is_zero() && zero_encoding == 0
                    );
                }
            }
        }
    }
}
