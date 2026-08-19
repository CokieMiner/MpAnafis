//! Property tests for balanced Toom-8 and unbalanced Toom-8.5.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{Multiplication, Schoolbook, dispatch::Widths};

fn balanced_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 8),
            proptest::collection::vec(any::<Limb>(), 8),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 65),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 129),
            proptest::collection::vec(any::<Limb>(), 125),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

fn half_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 9),
            proptest::collection::vec(any::<Limb>(), 8),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 73),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 145),
            proptest::collection::vec(any::<Limb>(), 129),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 289),
        ),
    ]
}

fn square_operands() -> impl Strategy<Value = Vec<Limb>> {
    prop_oneof![
        proptest::collection::vec(any::<Limb>(), 8),
        proptest::collection::vec(any::<Limb>(), 57),
        proptest::collection::vec(any::<Limb>(), 65),
        proptest::collection::vec(any::<Limb>(), 129),
        proptest::collection::vec(any::<Limb>(), 257),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_toom8_matches_basecase((a, b) in balanced_operands()) {
        prop_assert!(Widths::new(a.len(), b.len()).toom8_balanced());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom8_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom8_half_matches_basecase((a, b) in half_operands()) {
        prop_assert!(!Widths::new(a.len(), b.len()).toom8_balanced());
        prop_assert!(Widths::new(a.len(), b.len()).toom8_half_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom8_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom8_square_matches_basecase(a in square_operands()) {
        prop_assert!(Multiplication::operand_has_eight_parts(a.len()));
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![0; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::toom8_sqr_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::sqr(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}
