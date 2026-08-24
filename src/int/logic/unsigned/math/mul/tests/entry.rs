//! Property tests for owned multiplication and limb-dispatch entry points.

use alloc::{vec, vec::Vec};

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

#[test]
fn reusable_scratch_never_depends_on_previous_workspace_contents() {
    let mut scratch = MulScratch::default();
    for len in [64_usize, 288, 800, 2_048, 2_976, 512] {
        let left: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(0x9e37_79b9).wrapping_add(3) | 1)
            .collect();
        let right: Vec<Limb> = (0..len)
            .map(|index| index.wrapping_mul(0x85eb_ca6b).wrapping_add(5) | 1)
            .collect();
        let mut expected = vec![Limb::MAX; len.wrapping_mul(2)];
        Schoolbook::mul(&mut expected, &left, &right);

        scratch.buf.fill(Limb::MAX);
        let mut actual = vec![Limb::MIN; len.wrapping_mul(2)];
        Multiplication::mul_limbs_with_scratch(&left, &right, &mut actual, &mut scratch);
        assert_eq!(
            actual, expected,
            "dirty scratch changed the {len}-limb product"
        );
    }
}
