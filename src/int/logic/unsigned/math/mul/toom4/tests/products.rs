//! Arbitrary and maximal limbs over balanced, partial, and recursive splits.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::mul::Schoolbook;

use super::super::{Limb, Multiplication, Toom4};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 16 }))]

    #[test]
    fn products_and_squares_cover_partial_parts_carries_and_dirty_reuse(
        left_words in collection::vec(prop_oneof![3 => Just(Limb::MAX), 7 => any::<Limb>()], if cfg!(miri) { 20 } else { 2049 }),
        right_words in collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], if cfg!(miri) { 20 } else { 2049 }),
        left_width in 4_usize..=if cfg!(miri) { 20 } else { 2049 },
        right_width in 4_usize..=if cfg!(miri) { 20 } else { 2049 },
        guard_width in 320_usize..398,
    ) {
        for (len_a, len_b) in [(4, 4), (17, 17), (71, 71), (129, 129), (257, 257), (guard_width, guard_width), (1025, 1024), (left_width, right_width)] {
            if len_a > left_words.len() || len_b > right_words.len() { continue; }
            let left = left_words.get(..len_a).expect("bounded operand");
            let right = right_words.get(..len_b).expect("bounded operand");
            let width = len_a.checked_add(len_b).expect("test product fits");
            let mut expected = vec![0; width];
            Schoolbook::mul(&mut expected, left, right);
            let mut output = vec![Limb::MAX; width];
            let mut scratch = vec![Limb::MAX; Multiplication::toom4_mul_scratch_len(len_a, len_b)];
            for (a, b) in [(left, right), (right, left)] {
                for _ in 0..2 {
                    Toom4::mul(&mut output, a, b, &mut scratch);
                    prop_assert_eq!(&output, &expected);
                }
            }
            let square_width = len_a.checked_mul(2).expect("test square fits");
            let distinct = left.to_vec();
            let mut expected_square = vec![0; square_width];
            Schoolbook::mul(&mut expected_square, left, &distinct);
            let mut square = vec![Limb::MAX; square_width];
            let mut square_scratch = vec![Limb::MAX; Multiplication::toom4_sqr_scratch_len(len_a)];
            Toom4::sqr(&mut square, left, &mut square_scratch);
            prop_assert_eq!(&square, &expected_square);
            scratch.resize(Multiplication::toom4_mul_scratch_len(len_a, len_a), Limb::MAX);
            Toom4::mul(&mut square, left, left, &mut scratch);
            prop_assert_eq!(square, expected_square);
        }
    }
}
