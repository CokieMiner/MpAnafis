//! Native integer and byte-encoding round-trip properties.

use super::*;

proptest! {
    #[test]
    fn prop_native_conversions_roundtrip(u: u64, i: i64) {
        let au = MpUint::from(u);
        prop_assert_eq!(u64::try_from(au).expect("u64"), u);
        let ai = MpInt::from(i);
        prop_assert_eq!(i64::try_from(ai).expect("i64"), i);

        let small_u: u32 = (u & 0xFFFF_FFFF) as u32;
        let uint_32 = MpUint::from(small_u);
        prop_assert_eq!(u32::try_from(uint_32).expect("u32"), small_u);
        let small_i: i32 = (i & 0x7FFF_FFFF) as i32;
        let int_32 = MpInt::from(small_i);
        prop_assert_eq!(i32::try_from(int_32).expect("i32"), small_i);
    }
}

proptest! {
    #[test]
    fn prop_byte_serialization_roundtrip(u: u64, i: i64) {
        let au = MpUint::from(u);
        let le_bytes_u = au.to_le_bytes();
        prop_assert!(MpUint::from_le_bytes(&le_bytes_u) == au, "from_le_bytes");
        let be_bytes_u = au.to_be_bytes();
        prop_assert_eq!(MpUint::from_be_bytes(&be_bytes_u), au);

        let ai = MpInt::from(i);
        let le_bytes_i = ai.to_le_bytes();
        prop_assert!(MpInt::from_le_bytes(&le_bytes_i) == ai, "from_le_bytes");
        let be_bytes_i = ai.to_be_bytes();
        prop_assert_eq!(MpInt::from_be_bytes(&be_bytes_i), ai);
    }
}
