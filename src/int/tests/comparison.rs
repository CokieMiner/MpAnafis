//! Ordering and numeric-classification properties.

use super::*;

proptest! {
    #[test]
    fn prop_signed_cmp_antisymmetry(a in strategies::int(16), b in strategies::int(16)) {
        let cmp_ab = a.cmp(&b);
        let cmp_ba = b.cmp(&a);
        prop_assert_eq!(cmp_ab, cmp_ba.reverse(), "cmp anti-symmetry");
    }
}

proptest! {
    #[test]
    fn prop_signed_cmp_transitivity(a in strategies::int(8), b in strategies::int(8), c in strategies::int(8)) {
        if a < b && b < c { prop_assert!(a < c, "cmp transitivity"); }
        if a > b && b > c { prop_assert!(a > c, "cmp transitivity"); }
    }
}

proptest! {
    #[test]
    fn prop_signed_cmp_sign_inversion(a in strategies::int(16), b in strategies::int(16)) {
        let cmp_ab = a.cmp(&b);
        let cmp_neg_ba = (-&b).cmp(&(-&a));
        prop_assert_eq!(cmp_ab, cmp_neg_ba, "cmp sign inversion");
    }
}

proptest! {
    #[test]
    fn prop_numeric_properties(a in strategies::uint(8)) {
        let is_even_mod = (&a % MpUint::from(2_u8)).is_zero();
        prop_assert_eq!(a.is_even(), is_even_mod);
        prop_assert_eq!(a.is_odd(), !is_even_mod);

        prop_assert_eq!(a.significant_bits(), a.value.significant_bits());
        let i_pos = MpInt::from(a.clone());
        prop_assert_eq!(i_pos.significant_bits(), a.significant_bits());
        let i_neg = -i_pos;
        prop_assert_eq!(i_neg.significant_bits(), a.significant_bits());

        let is_pow2 = a.count_ones() == 1;
        prop_assert_eq!(a.is_power_of_two(), is_pow2);

        if !a.is_zero() {
            let next_pow2 = a.checked_next_power_of_two().expect("pow2");
            prop_assert!(next_pow2.is_power_of_two());
            prop_assert!(next_pow2 >= a);
        }
    }
}

proptest! {
    #[test]
    fn prop_new_public_apis_cmp(c1 in strategies::uint(16), c2 in strategies::uint(16), c3 in strategies::uint(16)) {
        let min_val = MpUint::min(c1.clone(), c2.clone());
        let max_val = MpUint::max(c1.clone(), c2.clone());
        let _clamp_val = MpUint::clamp(
            c1.clone(),
            MpUint::min(c2.clone(), c3.clone()),
            MpUint::max(c2.clone(), c3),
        );
        if c1 < c2 { prop_assert_eq!(min_val, c1); prop_assert_eq!(max_val, c2); }
        else { prop_assert_eq!(min_val, c2); prop_assert_eq!(max_val, c1); }
    }
}
