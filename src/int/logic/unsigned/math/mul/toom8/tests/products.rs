//! Balanced, nine-by-eight, and fallback products over exact reused scratch.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::mul::Schoolbook;

use super::super::{Limb, Multiplication, Toom8};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 8 }))]

    #[test]
    fn products_and_squares_cover_shapes_aliasing_and_dirty_guards(
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 17 } else { 400 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 17 } else { 400 }),
        left_width in 8_usize..=if cfg!(miri) { 17 } else { 400 },
        right_width in 8_usize..=if cfg!(miri) { 17 } else { 400 },
    ) {
        for (len_a, len_b) in [
            (8, 8), (9, 8), (16, 8), (17, 16), (57, 57), (65, 65), (129, 125),
            (257, 257), (73, 65), (145, 129), (257, 289), (128, 128), (left_width, right_width),
        ] {
            if len_a > left_words.len() || len_b > right_words.len() { continue; }
            let left = left_words.get(..len_a).expect("bounded operand");
            let right = right_words.get(..len_b).expect("bounded operand");
            let width = len_a.checked_add(len_b).expect("test product fits");
            let mut expected = vec![0; width];
            Schoolbook::mul(&mut expected, left, right);
            let mut output = vec![Limb::MAX; width.checked_add(3).expect("guards fit")];
            let mut scratch = vec![Limb::MAX; Multiplication::toom8_mul_scratch_len(len_a, len_b)];
            for (a, b) in [(left, right), (right, left)] {
                for _ in 0..2 {
                    let (product, guards) = output.split_at_mut(width);
                    product.fill(Limb::MAX);
                    Toom8::mul(product, a, b, &mut scratch);
                    prop_assert_eq!(&*product, expected.as_slice());
                    prop_assert_eq!(&*guards, &[Limb::MAX; 3]);
                }
            }
            let square_width = len_a.checked_mul(2).expect("test square fits");
            let distinct = left.to_vec();
            let mut expected_square = vec![0; square_width];
            Schoolbook::mul(&mut expected_square, left, &distinct);
            let mut square = vec![Limb::MAX; square_width];
            let mut square_scratch = vec![Limb::MAX; Multiplication::toom8_sqr_scratch_len(len_a)];
            Toom8::sqr(&mut square, left, &mut square_scratch);
            prop_assert_eq!(&square, &expected_square);
            scratch.resize(Multiplication::toom8_mul_scratch_len(len_a, len_a), Limb::MAX);
            Toom8::mul(&mut square, left, left, &mut scratch);
            prop_assert_eq!(square, expected_square);
        }
    }
}
