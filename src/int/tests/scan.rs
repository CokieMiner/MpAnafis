//! Bit scanning, range extraction, and convenience-wrapper properties.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "scan operators are the API under test"
)]

use core::ops::*;

use proptest::prelude::*;

use super::strategies;
use crate::int::api::MpUint;

proptest! {
    #[test]
    fn prop_bitwise_scan_zeros(a in strategies::uint(32), trailing_shift in 0_usize..=60) {
        let shifted = Shl::shl(a, trailing_shift);

        if shifted.value.is_zero() {
            prop_assert_eq!(shifted.trailing_zeros(), 0);
            prop_assert_eq!(shifted.find_first_set_bit(), None);
        } else {
            let tz = shifted.trailing_zeros();
            prop_assert!(tz >= trailing_shift);
            let fs = shifted.find_first_set_bit().expect("is not zero");
            prop_assert_eq!(fs, tz);
            prop_assert!(shifted.get_bit(fs));
        }
    }
}

proptest! {
    #[test]
    fn prop_bitwise_scan_ones(
        a in strategies::uint(32),
        trailing_ones_count in 0_usize..=60,
    ) {
        let mask = MpUint::one().shl(trailing_ones_count) - MpUint::one();
        let masked = BitOr::bitor(a, mask);

        if !masked.value.is_zero() && trailing_ones_count > 0 {
            let to = masked.trailing_ones();
            prop_assert!(to >= trailing_ones_count);
        }
    }
}

proptest! {
    #[test]
    fn prop_bitwise_scan_edge_cases(a in strategies::uint(4)) {
        if a.is_zero() {
            prop_assert_eq!(a.trailing_zeros(), 0);
            prop_assert_eq!(a.trailing_ones(), 0);
            prop_assert_eq!(a.leading_zeros(), None);
            prop_assert_eq!(a.find_first_set_bit(), None);
            prop_assert_eq!(a.count_ones(), 0);
        }
        if a == MpUint::one() {
            prop_assert_eq!(a.trailing_zeros(), 0);
            prop_assert_eq!(a.trailing_ones(), 1);
            prop_assert_eq!(a.find_first_set_bit(), Some(0));
            prop_assert_eq!(a.count_ones(), 1);
        }
    }
}

proptest! {
    #[test]
    fn prop_bitwise_convenience_wrappers(
        bit in 0_usize..=300,
        ua in strategies::uint(32),
        ia in strategies::int(32),
    ) {
        prop_assert_eq!(ua.test_bit(bit), ua.get_bit(bit));
        let ua_set = ua.set_bit(bit);
        prop_assert!(ua_set.test_bit(bit));
        prop_assert_eq!(ua_set.set_bit(bit), ua_set);

        let ua_clear = ua.clear_bit(bit);
        prop_assert!(!ua_clear.test_bit(bit));
        prop_assert_eq!(ua_clear.clear_bit(bit), ua_clear);

        let ua_toggle = ua.toggle_bit(bit);
        prop_assert_eq!(ua_toggle.test_bit(bit), !ua.test_bit(bit));
        prop_assert_eq!(ua_toggle.toggle_bit(bit), ua);

        prop_assert_eq!(ia.test_bit(bit), ia.get_bit(bit));
        let ia_set = ia.set_bit(bit);
        prop_assert!(ia_set.test_bit(bit));
        prop_assert_eq!(ia_set.set_bit(bit), ia_set);

        let ia_clear = ia.clear_bit(bit);
        prop_assert!(!ia_clear.test_bit(bit));
        prop_assert_eq!(ia_clear.clear_bit(bit), ia_clear);

        let ia_toggle = ia.toggle_bit(bit);
        prop_assert_eq!(ia_toggle.test_bit(bit), !ia.test_bit(bit));
        prop_assert_eq!(ia_toggle.toggle_bit(bit), ia);
    }
}

proptest! {
    #[test]
    fn prop_bitwise_int_tc_methods(
        val in strategies::int(32),
        from in 0_usize..=128,
        len in 1_usize..=64,
    ) {
        let to = from + len;
        let slice = val.bit_range(from, to);
        for i in 0..len {
            let si = slice.get_bit(i);
            let vi = val.get_bit(from + i);
            prop_assert_eq!(si, vi);
        }
    }
}
