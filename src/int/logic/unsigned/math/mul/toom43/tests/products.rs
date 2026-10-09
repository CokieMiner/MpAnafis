//! Signed evaluations, maximal carries, both orders, and dirty workspace reuse.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::mul::Schoolbook;

use super::super::{Limb, Multiplication, Toom43, Widths};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 24 }))]

    #[test]
    fn complete_products_cover_admitted_parts_orders_and_maximal_carries(
        larger in if cfg!(miri) { 9_usize..25 } else { 64_usize..400 },
        ratio_seed in 0_usize..1000,
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 24 } else { 400 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 24 } else { 400 }),
    ) {
        let split = larger.div_ceil(4);
        let smaller = split.checked_mul(2).and_then(|low| low.checked_add(1)).and_then(|low| low.checked_add(ratio_seed.rem_euclid(split))).expect("test ratio fits");
        prop_assume!(Widths::new(larger, smaller).toom43_suitable());
        let maximal_left = vec![Limb::MAX; larger];
        let maximal_right = vec![Limb::MAX; smaller];
        for (left, right) in [
            (left_words.get(..larger).expect("bounded operand"), right_words.get(..smaller).expect("bounded operand")),
            (maximal_left.as_slice(), maximal_right.as_slice()),
        ] {
            let width = larger.checked_add(smaller).expect("test product fits");
            let mut expected = vec![0; width];
            Schoolbook::mul(&mut expected, left, right);
            let mut actual = vec![Limb::MAX; width];
            let mut scratch = vec![Limb::MAX; Multiplication::toom43_mul_scratch_len(larger, smaller)];
            for (a, b) in [(left, right), (right, left)] {
                for _ in 0..2 {
                    actual.fill(Limb::MAX);
                    Toom43::mul(&mut actual, a, b, &mut scratch);
                    prop_assert_eq!(&actual, &expected);
                }
            }
        }
    }
}
