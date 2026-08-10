use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::{KARATSUBA_THRESHOLD, Schoolbook};

fn lopsided_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>, Limb)> {
    let minimum_len = KARATSUBA_THRESHOLD.max(2);
    let maximum_len = minimum_len.saturating_add(24);
    (minimum_len..=maximum_len, 8_usize..=12)
        .prop_flat_map(|(smaller_len, full_blocks)| {
            (Just(smaller_len), Just(full_blocks), 0_usize..smaller_len)
        })
        .prop_flat_map(|(smaller_len, full_blocks, tail_len)| {
            let larger_len = smaller_len
                .saturating_mul(full_blocks)
                .saturating_add(tail_len);
            (
                proptest::collection::vec(any::<Limb>(), larger_len),
                proptest::collection::vec(any::<Limb>(), smaller_len),
                any::<Limb>(),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_lopsided_dispatch_matches_basecase_from_dirty_output(
        (larger, smaller, dirty_limb) in lopsided_operands(),
    ) {
        let product_len = larger.len().wrapping_add(smaller.len());
        let mut expected = vec![Limb::MIN; product_len];
        Schoolbook::mul(&mut expected, &larger, &smaller);

        let scratch_len = Multiplication::required_scratch(larger.len(), smaller.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        let mut actual = vec![dirty_limb; product_len];
        Multiplication::mul_limbs_with_slice_scratch(
            &larger,
            &smaller,
            &mut actual,
            &mut scratch,
        );
        prop_assert_eq!(&actual, &expected);

        actual.fill(dirty_limb.wrapping_add(1));
        Multiplication::mul_limbs_with_slice_scratch(
            &smaller,
            &larger,
            &mut actual,
            &mut scratch,
        );
        prop_assert_eq!(&actual, &expected);

        let forced_block_len = smaller
            .len()
            .div_ceil(2)
            // SAFETY: smaller.len() > 0 by test precondition.
            .wrapping_add(unsafe { dirty_limb.checked_rem(smaller.len()).unwrap_unchecked() });
        let forced_scratch_len =
            Lopsided::mul_forced_scratch_len(larger.len(), smaller.len(), forced_block_len);
        let mut forced_scratch = vec![Limb::MIN; forced_scratch_len];
        actual.fill(dirty_limb.wrapping_add(2));
        Lopsided::mul_forced(
            &mut actual,
            &larger,
            &smaller,
            &mut forced_scratch,
            forced_block_len,
        );
        prop_assert_eq!(actual, expected);
    }
}
