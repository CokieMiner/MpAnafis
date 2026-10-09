//! Bit scans, range extraction, and bit-update identities on large values.

use proptest::prelude::{prop_assert, prop_assert_eq, proptest};

use crate::{MpUint, int::tests::strategies};

proptest! {
    #[test]
    fn scans_match_byte_counts_and_shifted_suffixes(
        input in strategies::uint(32), suffix in 0_usize..=60,
    ) {
        for value in [input.clone(), &input << suffix, &input | ((MpUint::one() << suffix) - MpUint::one()), MpUint::zero(), MpUint::one()] {
            let bytes = value.to_le_bytes();
            let zeros = bytes.iter().enumerate().find(|(_, byte)| **byte != 0)
                .map_or(0, |(index, byte)| index * 8 + byte.trailing_zeros() as usize);
            let first_non_full = bytes.iter().enumerate().find(|(_, byte)| **byte != u8::MAX);
            let ones = first_non_full.map_or(bytes.len() * 8, |(index, byte)| index * 8 + byte.trailing_ones() as usize);
            let set_bits: usize = bytes.iter().map(|byte| byte.count_ones() as usize).sum();
            prop_assert_eq!(value.trailing_zeros(), zeros);
            prop_assert_eq!(value.trailing_ones(), ones);
            prop_assert_eq!(value.count_ones(), set_bits);
            prop_assert_eq!(value.find_first_set_bit(), (!value.is_zero()).then_some(zeros));
            prop_assert_eq!(value.leading_zeros(), None);
            if !value.is_zero() { prop_assert!(value.get_bit(zeros)); }
        }
    }

    #[test]
    fn bit_updates_and_ranges_preserve_twos_complement_bits(
        bit in 0_usize..=300, from in 0_usize..=128, len in 1_usize..=64,
        unsigned in strategies::uint(32), signed in strategies::int(32),
    ) {
        prop_assert_eq!(unsigned.test_bit(bit), unsigned.get_bit(bit));
        let set = unsigned.set_bit(bit);
        prop_assert!(set.test_bit(bit));
        prop_assert_eq!(set.set_bit(bit), set);
        let clear = unsigned.clear_bit(bit);
        prop_assert!(!clear.test_bit(bit));
        prop_assert_eq!(clear.clear_bit(bit), clear);
        let toggle = unsigned.toggle_bit(bit);
        prop_assert_eq!(toggle.test_bit(bit), !unsigned.test_bit(bit));
        prop_assert_eq!(&toggle.toggle_bit(bit), &unsigned);

        prop_assert_eq!(signed.test_bit(bit), signed.get_bit(bit));
        let signed_set = signed.set_bit(bit);
        prop_assert!(signed_set.test_bit(bit));
        prop_assert_eq!(signed_set.set_bit(bit), signed_set);
        let signed_clear = signed.clear_bit(bit);
        prop_assert!(!signed_clear.test_bit(bit));
        prop_assert_eq!(signed_clear.clear_bit(bit), signed_clear);
        let signed_toggle = signed.toggle_bit(bit);
        prop_assert_eq!(signed_toggle.test_bit(bit), !signed.test_bit(bit));
        prop_assert_eq!(&signed_toggle.toggle_bit(bit), &signed);

        let unsigned_slice = unsigned.bit_range(from, from + len);
        let signed_slice = signed.bit_range(from, from + len);
        for index in 0..len {
            prop_assert_eq!(unsigned_slice.get_bit(index), unsigned.get_bit(from + index));
            prop_assert_eq!(signed_slice.get_bit(index), signed.get_bit(from + index));
        }
    }
}
