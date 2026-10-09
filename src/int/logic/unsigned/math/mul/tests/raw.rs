//! Limb entries, input aliasing, and reusable dirty scratch.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::int::logic::unsigned::math::mul::{Limb, MulScratch, Multiplication, Schoolbook};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 24 }))]

    #[test]
    fn limb_entries_match_schoolbook_and_zero_empty_products(
        left in prop::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 129 }),
        right in prop::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 129 }),
    ) {
        let width = left.len().checked_add(right.len()).expect("product fits");
        let mut expected = vec![0; width];
        Schoolbook::mul(&mut expected, &left, &right);
        let mut pool = MulScratch::default();
        let mut fixed = vec![Limb::MAX; Multiplication::required_scratch(left.len(), right.len())];
        for _ in 0..2 {
            let mut actual = vec![Limb::MAX; width];
            pool.buf.fill(Limb::MAX);
            Multiplication::mul_limbs_with_scratch(&left, &right, &mut actual, &mut pool);
            prop_assert_eq!(&actual, &expected);
            fixed.fill(Limb::MAX);
            actual.fill(Limb::MAX);
            Multiplication::mul_limbs_with_slice_scratch(&left, &right, &mut actual, &mut fixed);
            prop_assert_eq!(&actual, &expected);
        }
        let duplicate = left.clone();
        let square_width = left.len().checked_mul(2).expect("square fits");
        let mut square = vec![0; square_width];
        Schoolbook::mul(&mut square, &left, &duplicate);
        let mut actual = vec![Limb::MAX; square_width];
        let mut square_scratch = vec![Limb::MAX; Multiplication::required_scratch(left.len(), left.len())];
        Multiplication::mul_limbs_with_slice_scratch(&left, &left, &mut actual, &mut square_scratch);
        prop_assert_eq!(&actual, &square);
        Multiplication::sqr_limbs_with_scratch(&left, &mut actual, &mut pool);
        prop_assert_eq!(actual, square);
    }
}

#[cfg_attr(
    miri,
    ignore = "wide aliased products cover square and product crossover differences under native execution"
)]
#[test]
fn aliased_operands_accept_multiplication_sized_scratch() {
    for len in [
        27_usize, 28, 29, 186, 187, 188, 319, 320, 321, 464, 465, 466, 468, 469, 470, 2049,
    ] {
        let operand = vec![Limb::MAX; len];
        let width = len.checked_mul(2).expect("square fits");
        let mut expected = vec![0; width];
        Schoolbook::sqr(&mut expected, &operand);
        let mut actual = vec![Limb::MAX; width];
        let mut scratch = vec![Limb::MAX; Multiplication::required_scratch(len, len)];
        Multiplication::mul_limbs_with_slice_scratch(&operand, &operand, &mut actual, &mut scratch);
        assert_eq!(actual, expected, "aliased {len}-limb operands");
    }
}

#[cfg_attr(
    miri,
    ignore = "dirty workspace reuse spans transform and multi-thousand-limb crossover products"
)]
#[test]
fn reusable_scratch_spans_wide_plans_and_shrinking_requests() {
    let mut scratch = MulScratch::default();
    for len in [64_usize, 288, 800, 2_048, 2_976, 512] {
        let left: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(0x9e37_79b9).wrapping_add(3) | 1)
            .collect();
        let right: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(0x85eb_ca6b).wrapping_add(5) | 1)
            .collect();
        let width = len.checked_mul(2).expect("product fits");
        let mut expected = vec![0; width];
        Schoolbook::mul(&mut expected, &left, &right);
        scratch.buf.fill(Limb::MAX);
        let mut actual = vec![0; width];
        Multiplication::mul_limbs_with_scratch(&left, &right, &mut actual, &mut scratch);
        assert_eq!(actual, expected, "dirty scratch at {len} limbs");
    }
}
