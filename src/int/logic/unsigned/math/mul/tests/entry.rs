//! Property tests for owned multiplication and limb-dispatch entry points.

use alloc::vec;

use proptest::prelude::*;

use super::*;

proptest! {
    #[test]
    fn prop_in_place_matches_write_first(
        a_limbs in proptest::collection::vec(any::<Limb>(), 0..=16),
        b_limbs in proptest::collection::vec(any::<Limb>(), 0..=16),
    ) {
        let a = InternalMpUint::from_limbs(a_limbs);
        let b = InternalMpUint::from_limbs(b_limbs);
        let expected = a.mul(&b);
        let raw_len = a.limbs().len().wrapping_add(b.limbs().len());
        let mut raw = vec![Limb::MAX; raw_len];
        let mut scratch = MulScratch::default();
        Multiplication::mul_limbs_with_scratch(a.limbs(), b.limbs(), &mut raw, &mut scratch);
        let raw_active = raw
            .iter()
            .rposition(|limb| *limb != 0)
            .map_or(0, |index| index.wrapping_add(1));
        prop_assert_eq!(raw.get(..raw_active), Some(expected.limbs()));

        let mut actual = a.clone();
        actual.mul_assign(&b);
        prop_assert_eq!(actual.limbs(), expected.limbs());
    }
}
