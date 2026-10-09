//! Forced and dispatched Karatsuba with arbitrary limbs and reused storage.

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use super::super::{Karatsuba, Limb, Multiplication, Schoolbook};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 12 }))]

    #[test]
    fn products_and_squares_cover_shapes_aliasing_and_dirty_guards(
        left_words in collection::vec(any::<Limb>(), if cfg!(miri) { 48 } else { 257 }),
        right_words in collection::vec(any::<Limb>(), if cfg!(miri) { 48 } else { 257 }),
        random_left in 2_usize..=if cfg!(miri) { 24 } else { 128 },
        random_right in 2_usize..=if cfg!(miri) { 24 } else { 128 },
        dirty in any::<Limb>(),
    ) {
        for (left_width, right_width) in [
            (2, 2), (3, 3), (20, 20), (23, 23), (24, 24), (32, 32), (48, 48),
            (37, 37), (73, 41), (257, 257), (23, 22), (32, 31), (64, 48),
            (73, 72), (107, 106), (random_left, random_right),
        ] {
            if left_width > left_words.len() || right_width > right_words.len() { continue; }
            let left = left_words.get(..left_width).expect("bounded operand");
            let right = right_words.get(..right_width).expect("bounded operand");
            check_paths(left, right, dirty);
            check_paths(right, left, dirty);
            check_paths(left, left, dirty);
        }
    }
}

fn check_paths(left: &[Limb], right: &[Limb], dirty: Limb) {
    let width = left
        .len()
        .checked_add(right.len())
        .expect("test product fits");
    let mut expected = vec![0; width];
    let distinct_right = right.to_vec();
    Schoolbook::mul(&mut expected, left, &distinct_right);
    let mut actual = vec![dirty; width.checked_add(3).expect("guards fit")];
    let mut scratch =
        vec![Limb::MAX; Multiplication::karatsuba_mul_scratch_len(left.len(), right.len())];
    for _ in 0..2 {
        Karatsuba::mul(&mut actual, left, right, &mut scratch);
        let (forced_product, forced_guards) = actual.split_at(width);
        assert_eq!(forced_product, expected);
        assert_eq!(forced_guards, &[dirty; 3]);
        actual.fill(dirty);
        Karatsuba::dispatch_mul(&mut actual, left, right, &mut scratch);
        let (dispatched_product, dispatched_guards) = actual.split_at(width);
        assert_eq!(dispatched_product, expected);
        assert_eq!(dispatched_guards, &[dirty; 3]);
    }
    let square_width = left.len().checked_mul(2).expect("test square fits");
    expected.resize(square_width, 0);
    let distinct_left: Vec<_> = left.to_vec();
    Schoolbook::mul(&mut expected, left, &distinct_left);
    actual.resize(square_width.checked_add(3).expect("guards fit"), dirty);
    let mut square_scratch = vec![Limb::MAX; Multiplication::karatsuba_sqr_scratch_len(left.len())];
    for _ in 0..2 {
        actual.fill(dirty);
        Karatsuba::sqr(&mut actual, left, &mut square_scratch);
        let (forced_square, forced_guards) = actual.split_at(square_width);
        assert_eq!(forced_square, expected);
        assert_eq!(forced_guards, &[dirty; 3]);
        actual.fill(dirty);
        Karatsuba::dispatch_sqr(&mut actual, left, &mut square_scratch);
        let (dispatched_square, dispatched_guards) = actual.split_at(square_width);
        assert_eq!(dispatched_square, expected);
        assert_eq!(dispatched_guards, &[dirty; 3]);
    }
}
