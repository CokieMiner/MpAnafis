//! Significance, zero extension, and guarded copying.

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::InternalMpUint;

use super::super::{Limb, SharedEval};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn preparation_matches_normalized_magnitudes_and_preserves_guards(
        words in collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 48 }),
        narrow_length in 0_usize..=48,
        zero_suffix in 0_usize..=4,
    ) {
        let mut wide = words;
        wide.extend(core::iter::repeat_n(0, zero_suffix));
        let narrow = wide.get(..narrow_length.min(wide.len())).expect("bounded prefix");
        let wide_value = InternalMpUint::from_limbs_slice(&wide);
        let narrow_value = InternalMpUint::from_limbs_slice(narrow);
        prop_assert_eq!(SharedEval::active_len(&wide), wide_value.limbs().len());
        prop_assert_eq!(SharedEval::compare_with_zero_extension(&wide, narrow), wide_value.cmp(&narrow_value));

        let mut destination = vec![Limb::MAX; wide.len().checked_add(3).expect("guards fit")];
        SharedEval::copy_part(&mut destination, &wide);
        let (copied, guards) = destination.split_at(wide.len());
        prop_assert_eq!(copied, wide.as_slice());
        prop_assert_eq!(guards, &[0, 0, 0]);

        let smaller = vec![Limb::MAX; wide.len()];
        prop_assert_eq!(SharedEval::compare_with_zero_extension(&wide, &smaller), wide_value.cmp(&InternalMpUint::from_limbs_slice(&smaller)));
    }
}
