//! Root forcing, normal dispatch, unequal splits, and recursive widths.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::mul::Schoolbook;

use super::super::{Limb, Multiplication, Toom3};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 8 }))]

    #[test]
    fn products_and_squares_cover_forcing_dispatch_and_reused_guards(
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 24 } else { 1025 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 24 } else { 1025 }),
        left_width in 3_usize..=if cfg!(miri) { 24 } else { 1025 },
        right_width in 3_usize..=if cfg!(miri) { 24 } else { 1025 },
        dirty in any::<Limb>(),
    ) {
        for (len_a, len_b) in [(3, 3), (4, 3), (5, 5), (70, 70), (113, 109), (257, 257), (513, 512), (left_width, right_width)] {
            if len_a > left_words.len() || len_b > right_words.len() { continue; }
            let left = left_words.get(..len_a).expect("bounded operand");
            let right = right_words.get(..len_b).expect("bounded operand");
            let width = len_a.checked_add(len_b).expect("test product fits");
            let mut expected = vec![0; width];
            Schoolbook::mul(&mut expected, left, right);
            let mut output = vec![dirty; width.checked_add(3).expect("guards fit")];
            let forced_len = Multiplication::toom3_mul_scratch_len(len_a, len_b);
            let dispatch_len = Multiplication::toom3_dispatch_mul_scratch_len(len_a, len_b);
            let mut scratch = vec![Limb::MAX; forced_len.max(dispatch_len)];
            for (a, b) in [(left, right), (right, left)] {
                for _ in 0..2 {
                    // Toom-3 clears the supplied destination outside its endpoints.
                    let (product, guards) = output.split_at_mut(width);
                    Toom3::mul(product, a, b, scratch.get_mut(..forced_len).expect("forced workspace fits"));
                    prop_assert_eq!(&*product, expected.as_slice());
                    prop_assert_eq!(&*guards, &[dirty; 3]);
                    product.fill(dirty);
                    Toom3::dispatch_mul(product, a, b, scratch.get_mut(..dispatch_len).expect("dispatch workspace fits"));
                    prop_assert_eq!(&*product, expected.as_slice());
                }
            }
            let square_width = len_a.checked_mul(2).expect("test square fits");
            let separate_left = left.to_vec();
            let mut expected_square = vec![0; square_width];
            Schoolbook::mul(&mut expected_square, left, &separate_left);
            let mut square = vec![dirty; square_width];
            let forced_square_len = Multiplication::toom3_sqr_scratch_len(len_a);
            let dispatch_square_len = Multiplication::toom3_dispatch_sqr_scratch_len(len_a);
            let mut square_scratch = vec![Limb::MAX; forced_square_len.max(dispatch_square_len)];
            Toom3::sqr(&mut square, left, square_scratch.get_mut(..forced_square_len).expect("forced square workspace fits"));
            prop_assert_eq!(&square, &expected_square);
            square.fill(dirty);
            Toom3::dispatch_sqr(&mut square, left, square_scratch.get_mut(..dispatch_square_len).expect("dispatch square workspace fits"));
            prop_assert_eq!(&square, &expected_square);
            square.fill(dirty);
            scratch.resize(Multiplication::toom3_mul_scratch_len(len_a, len_a), Limb::MAX);
            Toom3::mul(&mut square, left, left, &mut scratch);
            prop_assert_eq!(square, expected_square);
        }
    }
}
