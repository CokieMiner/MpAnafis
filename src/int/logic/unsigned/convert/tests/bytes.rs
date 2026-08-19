use proptest::prelude::*;

use super::*;

proptest! {
    #[test]
    fn test_byte_roundtrip_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
    ) {
        let val = InternalMpUint::from_limbs(limbs);

        let le_bytes = val.to_le_bytes();
        let from_le = InternalMpUint::from_le_bytes(&le_bytes);
        prop_assert!(from_le == val);

        let be_bytes = val.to_be_bytes();
        let from_be = InternalMpUint::from_be_bytes(&be_bytes);
        prop_assert!(from_be == val);
    }
}
