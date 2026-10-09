//! Ordering laws and numeric classification across inline and heap magnitudes.

extern crate std;

use std::panic::catch_unwind;

use proptest::prelude::{prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

use super::strategies;

proptest! {
    #[test]
    fn ordering_and_classification_obey_numeric_laws(
        a in strategies::int(16), b in strategies::int(16), c in strategies::int(16),
        x in strategies::uint(16), y in strategies::uint(16), z in strategies::uint(16),
    ) {
        prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        prop_assert_eq!(a.cmp(&b), (-&b).cmp(&(-&a)));
        if a < b && b < c { prop_assert!(a < c); }
        if a > b && b > c { prop_assert!(a > c); }
        let signed_lower = MpInt::min(b.clone(), c.clone());
        let signed_upper = MpInt::max(b, c);
        let clamped_signed = MpInt::clamp(a.clone(), signed_lower.clone(), signed_upper.clone());
        prop_assert_eq!(&clamped_signed, if a < signed_lower { &signed_lower } else if a > signed_upper { &signed_upper } else { &a });
        prop_assert_eq!(a.signum(), if a.is_zero() { MpInt::zero() } else if a.is_negative() { -MpInt::one() } else { MpInt::one() });
        prop_assert_eq!(a.is_even(), a.unsigned_abs().is_even());
        prop_assert_eq!(a.is_odd(), a.unsigned_abs().is_odd());
        prop_assert_eq!(a.is_one(), a == MpInt::one());
        prop_assert_eq!(a.is_minus_one(), a == -MpInt::one());
        prop_assert_eq!(a.is_power_of_two(), a.is_positive() && a.unsigned_abs().is_power_of_two());
        prop_assert_eq!(x.cmp(&y), y.cmp(&x).reverse());
        if x < y && y < z { prop_assert!(x < z); }
        if x > y && y > z { prop_assert!(x > z); }
        let minimum = MpUint::min(x.clone(), y.clone());
        let maximum = MpUint::max(x.clone(), y.clone());
        prop_assert!(minimum <= x && minimum <= y);
        prop_assert!(maximum >= x && maximum >= y);
        let lower = MpUint::min(y.clone(), z.clone());
        let upper = MpUint::max(y, z);
        let clamped = MpUint::clamp(x.clone(), lower.clone(), upper.clone());
        prop_assert_eq!(&clamped, if x < lower { &lower } else if x > upper { &upper } else { &x });

        let even = (&x % MpUint::from(2_u8)).is_zero();
        prop_assert_eq!(x.is_even(), even);
        prop_assert_eq!(x.is_odd(), !even);
        let bytes = x.to_le_bytes();
        let bit_width = bytes.last().map_or(0, |last| (bytes.len() - 1) * 8 + (8 - last.leading_zeros() as usize));
        prop_assert_eq!(x.significant_bits(), bit_width);
        let positive = MpInt::from(x.clone());
        let negative = -&positive;
        prop_assert_eq!(positive.significant_bits(), bit_width);
        prop_assert_eq!(negative.significant_bits(), bit_width);
        prop_assert_eq!(positive.is_positive(), !x.is_zero());
        prop_assert_eq!(negative.is_negative(), !x.is_zero());
        prop_assert_eq!(x.is_power_of_two(), x.count_ones() == 1);
        prop_assert_eq!(x.is_one(), x == MpUint::one());
        if !x.is_zero() {
            let next = x.checked_next_power_of_two().expect("unlimited precision");
            prop_assert!(next.is_power_of_two() && next >= x);
            if next != x { prop_assert!((&next >> 1_usize) < x); }
        }
    }
}

#[test]
fn clamps_reject_reversed_bounds_for_both_integer_domains() {
    for input in [0_u8, 3, 5, 9, 255] {
        assert!(
            catch_unwind(|| MpUint::from(input).clamp(MpUint::from(9_u8), MpUint::from(3_u8)))
                .is_err()
        );
        assert!(
            catch_unwind(|| MpInt::from(input).clamp(MpInt::from(9_u8), MpInt::from(3_u8)))
                .is_err()
        );
    }
}
