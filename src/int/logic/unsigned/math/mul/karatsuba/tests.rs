//! Property tests for Karatsuba multiplication and squaring.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{Multiplication, Schoolbook};

fn operand_pairs() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 20),
            proptest::collection::vec(any::<Limb>(), 20),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 32),
            proptest::collection::vec(any::<Limb>(), 32),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 48),
            proptest::collection::vec(any::<Limb>(), 48),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 37),
            proptest::collection::vec(any::<Limb>(), 37),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 73),
            proptest::collection::vec(any::<Limb>(), 41),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

fn balanced_even_operand_pairs() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    (10_usize..=64).prop_flat_map(|half_len| {
        let operand_len = half_len.saturating_mul(2);
        (
            proptest::collection::vec(any::<Limb>(), operand_len),
            proptest::collection::vec(any::<Limb>(), operand_len),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn prop_karatsuba_matches_basecase((a, b) in operand_pairs()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::karatsuba_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![0; scratch_len];
        Karatsuba::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_forced_karatsuba_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 2..70),
        b in proptest::collection::vec(any::<Limb>(), 2..70),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::karatsuba_mul_forced_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Karatsuba::mul_forced(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_balanced_even_difference_matches_basecase(
        (a, b) in balanced_even_operand_pairs(),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::karatsuba_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![0; scratch_len];
        Karatsuba::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_karatsuba_square_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 20..258),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::karatsuba_sqr_scratch_len(a.len());
        let mut scratch = vec![0; scratch_len];
        Karatsuba::sqr(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_forced_karatsuba_square_matches_basecase(
        a in proptest::collection::vec(any::<Limb>(), 2..70),
    ) {
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::karatsuba_sqr_forced_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Karatsuba::sqr_forced(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}
