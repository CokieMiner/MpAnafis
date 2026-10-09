//! Root variants, exact-square normalization, and recursive division crossovers.

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert, prop_assert_eq, proptest,
};

use crate::int::logic::unsigned::math::{BURNIKEL_ZIEGLER_THRESHOLD, NEWTON_RAPHSON_THRESHOLD};

use super::super::{InternalMpUint, Limb};

#[test]
fn square_roots_preserve_sparse_dense_and_normalization_boundaries() {
    let one = InternalMpUint::one();
    let widths: alloc::vec::Vec<_> = if cfg!(miri) {
        alloc::vec![1, 4, 5]
    } else {
        (1..=24).chain([32]).collect()
    };
    for width in widths {
        let mut sparse = alloc::vec![0; width];
        *sparse.last_mut().expect("nonempty fixture") = Limb::MAX ^ 0x5555;
        for input in [
            InternalMpUint::from_limbs(sparse),
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX.div_euclid(3); width]),
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]),
        ] {
            check_square_root(&input);
        }
        let shifts: alloc::vec::Vec<_> = if cfg!(miri) {
            alloc::vec![0, 1, Limb::BITS >> 1, Limb::BITS - 1]
        } else {
            (0..Limb::BITS).collect()
        };
        for shift in shifts {
            let mut words = alloc::vec![Limb::MAX; width];
            *words.last_mut().expect("nonempty root") = 1 << shift;
            let root = InternalMpUint::from_limbs(words);
            let square = root.square();
            assert_eq!(square.sqrt_rem(), (root.clone(), InternalMpUint::zero()));
            assert_eq!(square.add(&one).sqrt_rem(), (root.clone(), one.clone()));
            let below = root.sub(&one);
            assert_eq!(square.sub(&one).sqrt_rem(), (below.clone(), below.shl(1)));
            assert_eq!(square.sub(&one).isqrt(), below);
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Configured division crossovers and 8192-limb recursive correction widths require native execution; guarded recursive root properties run under Miri."
)]
fn recursive_roots_cross_division_tiers_and_correction_geometries() {
    let one = InternalMpUint::one();
    for crossover in [BURNIKEL_ZIEGLER_THRESHOLD, NEWTON_RAPHSON_THRESHOLD] {
        for width in [crossover - 1, crossover, crossover + 1] {
            let root = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 2 * width]);
            let square = root.square();
            for (input, expected_root, expected_rem) in [
                (square.sub(&one), root.sub(&one), root.sub(&one).shl(1)),
                (square.clone(), root.clone(), InternalMpUint::zero()),
                (square.add(&root.shl(1)), root.clone(), root.shl(1)),
            ] {
                assert_eq!(input.isqrt(), expected_root);
                assert_eq!(input.sqrt_rem(), (expected_root, expected_rem));
            }
        }
    }
    for width in [8192, 8193, 8194, 8208] {
        let mut dense = alloc::vec![Limb::MAX.div_euclid(3); width];
        *dense.last_mut().expect("nonempty fixture") = Limb::MAX;
        check_square_root(&InternalMpUint::from_limbs(dense));
        check_square_root(&InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]));
    }
}

fn check_square_root(input: &InternalMpUint) {
    let root = input.isqrt();
    let (paired, remainder) = input.sqrt_rem();
    assert_eq!(paired, root);
    assert_eq!(&root.square().add(&remainder), input);
    assert!(remainder < root.shl(1).add(&InternalMpUint::one()));
    assert!(root.square() <= *input);
    assert!(*input < root.add(&InternalMpUint::one()).square());
    assert_eq!(input.is_perfect_square(), remainder.is_zero());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))]
    #[test]
    fn all_root_variants_preserve_exact_integer_brackets(
        words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 24 }),
        degree in 2_u32..=130,
    ) {
        let input = InternalMpUint::from_limbs(words);
        check_square_root(&input);
        let root = input.nth_root(degree);
        prop_assert!(root.pow(degree) <= input);
        prop_assert!(root.add(&InternalMpUint::one()).pow(degree) > input);
        prop_assert_eq!(input.nth_root(2), input.isqrt());
    }
}
