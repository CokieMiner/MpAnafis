use proptest::prelude::*;

use super::*;
use crate::int::types::Limb;

proptest! {
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest for signed bits"
    )]
    #[test]
    fn required_signed_bits_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
        is_positive in any::<bool>(),
    ) {
        let abs = InternalMpUint::from_limbs(limbs);
        let pos = is_positive || abs.is_zero();
        let val = InternalMpInt {
            abs,
            is_positive: pos,
        };
        let n = val.required_signed_bits_for_bounded_storage();
        prop_assert!(n >= 1);

        let tc = val.to_tc_bits(n);
        let restored = InternalMpInt::from_tc_bits(tc, n);
        prop_assert!(restored.abs == val.abs);
        prop_assert_eq!(restored.is_positive, val.is_positive);

        if n > 1 {
            let tc_minus = val.to_tc_bits(n - 1);
            let restored_minus = InternalMpInt::from_tc_bits(tc_minus, n - 1);
            prop_assert!(
                restored_minus.abs != val.abs
                    || restored_minus.is_positive != val.is_positive
            );
        }
    }
}
