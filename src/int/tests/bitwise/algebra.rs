//! Bitwise algebra, scanning, and unsigned byte permutation on large values.

use proptest::prelude::{prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, int::tests::strategies};

proptest! {
    #[test]
    fn bitwise_algebra_and_scanning_obey_integer_identities(
        a in strategies::int(16), b in strategies::int(16), c in strategies::int(16),
        x in strategies::uint(8), y in strategies::uint(8),
    ) {
        prop_assert_eq!(&(&a & &a), &a);
        prop_assert_eq!(&(&a | &a), &a);
        prop_assert_eq!(&a ^ &a, MpInt::zero());
        prop_assert_eq!(!&a, -&a - MpInt::one());
        prop_assert_eq!(&a & &b, &b & &a);
        prop_assert_eq!(&a | &b, &b | &a);
        prop_assert_eq!(&a ^ &b, &b ^ &a);
        prop_assert_eq!(&(&a ^ &b) ^ &c, &a ^ &(&b ^ &c));
        prop_assert_eq!(&(&a & &b) & &c, &a & &(&b & &c));
        prop_assert_eq!(&(&a | &b) | &c, &a | &(&b | &c));
        let intersection = &x & &y;
        let union = &x | &y;
        prop_assert_eq!(&(&intersection | &union), &union);
        prop_assert_eq!(&intersection & &union, intersection);
        prop_assert_eq!(&(&(&x ^ &y) ^ &y), &x);
        if let Some(first) = x.find_first_set_bit() {
            prop_assert_eq!(x.trailing_zeros(), first);
            prop_assert!(x.get_bit(first));
            prop_assert_eq!(x.find_next_set_bit(first + 1), x.clear_bit(first).find_first_set_bit());
        } else {
            prop_assert!(x.is_zero());
            prop_assert_eq!(x.trailing_zeros(), 0);
        }
        let odd = x | MpUint::one();
        prop_assert_eq!(odd.swap_bytes().swap_bytes(), odd);
    }
}
