//! Fixed carry chains and zero differences at split boundaries.

use alloc::{vec, vec::Vec};

use super::super::{Karatsuba, Limb, Multiplication, Schoolbook};

#[test]
fn carry_chains_cover_specializations_and_recursive_split_boundaries() {
    for len in [
        2_usize, 3, 19, 20, 21, 23, 24, 31, 32, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65,
        127, 128, 129,
    ] {
        if cfg!(miri) && len > 24 {
            continue;
        }
        let width = len.checked_mul(2).expect("test product fits");
        let mut product_scratch =
            vec![Limb::MAX; Multiplication::karatsuba_mul_scratch_len(len, len)];
        let mut square_scratch = vec![Limb::MAX; Multiplication::karatsuba_sqr_scratch_len(len)];
        let mut actual = vec![Limb::MAX; width];
        let mut expected = vec![0; width];
        for left_pattern in 0..4 {
            let left = carry_operand(len, left_pattern);
            let distinct_left = left.clone();
            Schoolbook::mul(&mut expected, &left, &distinct_left);
            Karatsuba::sqr(&mut actual, &left, &mut square_scratch);
            assert_eq!(actual, expected, "square {len}, pattern {left_pattern}");
            Karatsuba::mul(&mut actual, &left, &left, &mut product_scratch);
            assert_eq!(
                actual, expected,
                "aliased inputs {len}, pattern {left_pattern}"
            );
            for right_pattern in 0..4 {
                let right = carry_operand(len, right_pattern);
                Schoolbook::mul(&mut expected, &left, &right);
                Karatsuba::mul(&mut actual, &left, &right, &mut product_scratch);
                assert_eq!(
                    actual, expected,
                    "product {len}, patterns {left_pattern}/{right_pattern}"
                );
            }
        }
    }
}

fn carry_operand(len: usize, pattern: usize) -> Vec<Limb> {
    (0..len)
        .map(|index| match pattern {
            1 => Limb::MAX,
            2 if index.is_multiple_of(2) => Limb::MAX,
            3 if index >= len.div_ceil(2) => Limb::MAX,
            _ => 0,
        })
        .collect()
}
