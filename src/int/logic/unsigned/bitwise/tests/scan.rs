use proptest::prelude::*;

use super::*;

proptest! {
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest for unsigned bits"
    )]
    #[test]
    fn required_unsigned_bits_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
    ) {
        let val = InternalMpUint::from_limbs(limbs);
        let n = val.required_unsigned_bits_for_bounded_storage();
        let sig = val.significant_bits();

        if val.is_zero() {
            prop_assert_eq!(n, 1);
        } else {
            prop_assert_eq!(n, sig);
            let max_val = InternalMpUint::max_for_bits(n);
            prop_assert!(val <= max_val);
            if n > 1 {
                let max_minus = InternalMpUint::max_for_bits(n - 1);
                prop_assert!(val > max_minus);
            }
        }
    }
}
