//! Complete products and squares across all Toom-6 shape branches.

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::mul::{MulShape, Schoolbook};

use super::super::{Limb, Multiplication, Toom6, Widths};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 8 }))]

    #[test]
    fn products_and_squares_cover_balanced_half_and_fallback_shapes(
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 16 } else { 2049 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 16 } else { 2049 }),
        left_width in 6_usize..=if cfg!(miri) { 16 } else { 2049 },
        right_width in 6_usize..=if cfg!(miri) { 16 } else { 2049 },
        dirty in any::<Limb>(),
    ) {
        for (len_a, len_b) in [
            (6, 6), (7, 6), (14, 12), (37, 37), (113, 109), (257, 257),
            (70, 60), (113, 100), (221, 257), (1025, 1024), (left_width, right_width),
        ] {
            if len_a > left_words.len() || len_b > right_words.len() { continue; }
            check_paths(left_words.get(..len_a).expect("bounded operand"), right_words.get(..len_b).expect("bounded operand"), dirty);
        }
        let lengths: &[usize] = if cfg!(miri) { &[6, 7, 12, 14] } else { &[40, 61, 96, 113, 180, 257] };
        let mut branches = [false; 3];
        for &larger in lengths {
            for (numerator, denominator) in [(1_usize, 1_usize), (6, 7), (5, 6), (3, 4), (1, 2), (1, 3), (1, 6), (1, 12)] {
                let smaller = larger.checked_mul(numerator).expect("test ratio fits").div_euclid(denominator);
                if smaller < 6 { continue; }
                match Widths::new(larger, smaller).toom6_shape() {
                    Some(MulShape::Balanced) => *branches.get_mut(0).expect("balanced marker") = true,
                    Some(MulShape::Half) => *branches.get_mut(1).expect("half marker") = true,
                    None => *branches.get_mut(2).expect("fallback marker") = true,
                }
                let a: Vec<_> = (0..larger).map(|index| Limb::MAX.wrapping_sub(index.wrapping_mul(0x9E37))).collect();
                let b: Vec<_> = (0..smaller).map(|index| Limb::MAX.wrapping_sub(index.wrapping_mul(0x7F4A))).collect();
                check_paths(&a, &b, dirty);
            }
        }
        prop_assert!(branches.into_iter().all(|seen| seen), "balanced, half, and fallback branches must execute");
    }
}

fn check_paths(left: &[Limb], right: &[Limb], dirty: Limb) {
    let width = left
        .len()
        .checked_add(right.len())
        .expect("test product fits");
    let mut expected = vec![0; width];
    Schoolbook::mul(&mut expected, left, right);
    let mut output = vec![dirty; width.checked_add(3).expect("guards fit")];
    let mut scratch =
        vec![Limb::MAX; Multiplication::toom6_mul_scratch_len(left.len(), right.len())];
    for (a, b) in [(left, right), (right, left)] {
        for _ in 0..2 {
            let (product, guards) = output.split_at_mut(width);
            product.fill(dirty);
            Toom6::mul(product, a, b, &mut scratch);
            assert_eq!(product, expected);
            assert_eq!(guards, &[dirty; 3]);
        }
    }
    let square_width = left.len().checked_mul(2).expect("test square fits");
    let separate_left = left.to_vec();
    let mut expected_square = vec![0; square_width];
    Schoolbook::mul(&mut expected_square, left, &separate_left);
    let mut square = vec![dirty; square_width];
    let mut square_scratch = vec![Limb::MAX; Multiplication::toom6_sqr_scratch_len(left.len())];
    for _ in 0..2 {
        Toom6::sqr(&mut square, left, &mut square_scratch);
        assert_eq!(square, expected_square);
        scratch.resize(
            Multiplication::toom6_mul_scratch_len(left.len(), left.len()),
            Limb::MAX,
        );
        Toom6::mul(&mut square, left, left, &mut scratch);
        assert_eq!(square, expected_square);
    }
}
