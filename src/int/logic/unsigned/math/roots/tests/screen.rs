//! Square-screen residues against independent scalar modular arithmetic.

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::super::{InternalMpUint, Limb, Roots};

#[test]
fn residue_screens_preserve_carries_partial_groups_and_all_squares() {
    let check = |value: &InternalMpUint| {
        let expected = [256_u128, 9, 5, 7, 13, 17, 97, 241, 257, 673]
            .into_iter()
            .all(|modulus| {
                let residue = value.limbs().iter().rev().fold(0_u128, |acc, &limb| {
                    ((acc << Limb::BITS) | u128::try_from(limb).expect("limb fits u128"))
                        .rem_euclid(modulus)
                });
                (0..modulus).any(|root| root.pow(2).rem_euclid(modulus) == residue)
            });
        assert_eq!(Roots::may_be_square(value), expected);
        assert!(Roots::may_be_square(&value.square()));
        if !expected {
            assert!(!value.is_perfect_square());
        }
    };
    for &width in if cfg!(miri) {
        &[0, 1, 4, 5][..]
    } else {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 127, 128, 129][..]
    } {
        let mut words = alloc::vec![Limb::MAX; width];
        if let Some(low) = words.first_mut() {
            *low = 1;
        }
        check(&InternalMpUint::from_limbs(words));
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))
        .run(
            &collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 64 }),
            |words| {
                check(&InternalMpUint::from_limbs(words));
                Ok(())
            },
        )
        .expect("square-screen property");
}
