//! Newton lower-quotient residue identities and scalar correction boundaries.

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]

    #[test]
    fn block_residue_products_match_lower_quotient_bounds(
        mut denominator in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 8 } else { 260 }),
        mut quotient in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 260 }),
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << (Limb::BITS - 1);
        quotient.truncate(denominator.len());
        let divisor = InternalMpUint::from_limbs(denominator);
        let exact = InternalMpUint::from_limbs(quotient);
        let n = divisor.limbs().len();
        let width = n.checked_add(exact.limbs().len().max(1))
            .and_then(|len| len.checked_add(1)).expect("bounded guarded dividend");
        let mut scratch = DivScratch::default();
        for error in 0..=3 {
            let delta = InternalMpUint::from_limb(error);
            if exact < delta {
                continue;
            }
            let estimate = exact.sub(&delta);
            for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                let numerator = divisor.mul(&exact).add(&residue);
                let expected = divisor.mul(&delta).add(&residue);
                let mut input = numerator.limbs().to_vec();
                input.resize(width, 0);
                scratch.v_padded.clear();
                scratch.v_padded.push(Limb::MAX);
                scratch.v_padded.extend_from_slice(estimate.limbs());
                scratch.v_padded.push(0);
                let remainder = Division::newton_remainder::<false>(
                    &mut input,
                    divisor.limbs(),
                    1,
                    &mut scratch,
                );
                prop_assert_eq!(InternalMpUint::from_limbs_slice(remainder), expected);
            }
        }
    }
}

#[test]
fn scalar_residues_cross_limb_corrections_and_reuse_storage() {
    let mut scratch = DivScratch::default();
    for n in [1_usize, 2, 3, 31, 32, 33, 127, 128, 129, 257] {
        if cfg!(miri) && n > 33 {
            continue;
        }
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n]);
        for digit in [1, 2, Limb::MAX >> 1, Limb::MAX] {
            let estimate = InternalMpUint::from_limb(digit);
            for error in 0..=3 {
                let delta = InternalMpUint::from_limb(error);
                let exact = estimate.add(&delta);
                for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                    let numerator = divisor.mul(&exact).add(&residue);
                    let expected = divisor.mul(&delta).add(&residue);
                    let mut input = numerator.limbs().to_vec();
                    input.resize(input.len().max(n.checked_add(1).expect("residue guard")), 0);
                    scratch.v_padded.clear();
                    scratch.v_padded.push(digit);
                    scratch.v_padded.push(0);
                    let remainder = Division::newton_remainder::<false>(
                        &mut input,
                        divisor.limbs(),
                        0,
                        &mut scratch,
                    );
                    assert_eq!(
                        InternalMpUint::from_limbs_slice(remainder),
                        expected,
                        "scalar residue n={n}, digit={digit}, error={error}"
                    );
                }
            }
        }
    }
}
