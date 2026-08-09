//! Property tests for the Toom-Cook 3 tier.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{Multiplication, Schoolbook};

fn balanced_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 70),
            proptest::collection::vec(any::<Limb>(), 70),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 113),
            proptest::collection::vec(any::<Limb>(), 109),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_forced_toom3_matches_basecase((a, b) in balanced_operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom3_mul_forced_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom3::mul_forced(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom3_square_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 70..258),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::toom3_sqr_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom3::sqr(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_forced_toom3_square_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 3..258),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::toom3_sqr_forced_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom3::sqr_forced(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2))]

    #[test]
    fn prop_toom3_recursive_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 512..1026),
        b in proptest::collection::vec(any::<Limb>(), 512..1026),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom3_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom3::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}
