//! Native-reference rounding and heap-sized signed division contracts.

use proptest::prelude::{Just, Strategy, any, prop_assert_eq, prop_oneof, proptest};

use crate::{BoundedPrecision, MpInt};

proptest! {
    #[test]
    fn rounded_division_matches_native_reference(
        left in prop_oneof![Just(i64::MIN), Just(i64::MAX), any::<i64>()],
        right in any::<i64>().prop_filter("nonzero divisor", |value| *value != 0),
    ) {
        let dividend = i128::from(left);
        let divisor = i128::from(right);
        let euclid = dividend.div_euclid(divisor);
        let remainder = dividend.rem_euclid(divisor);
        let floor = if divisor < 0 && remainder != 0 {
            euclid.checked_sub(1).expect("i64 quotient fits i128")
        } else { euclid };
        let ceil = if divisor > 0 && remainder != 0 {
            euclid.checked_add(1).expect("i64 quotient fits i128")
        } else { euclid };
        let a = MpInt::from(left);
        let b = MpInt::from(right);
        prop_assert_eq!(a.checked_div(&b).and_then(|v| v.to_i128()), dividend.checked_div(divisor));
        prop_assert_eq!(a.checked_rem(&b).and_then(|v| v.to_i128()), Some(dividend % divisor));
        prop_assert_eq!(a.checked_div_euclid(&b).and_then(|value| value.to_i128()), Some(euclid));
        prop_assert_eq!(a.checked_div_floor(&b).and_then(|value| value.to_i128()), Some(floor));
        prop_assert_eq!(a.checked_div_ceil(&b).and_then(|value| value.to_i128()), Some(ceil));
        let (euclid_q, euclid_r) = a.div_rem_euclid(&b).expect("unlimited nonzero divisor");
        prop_assert_eq!(euclid_q.to_i128(), Some(euclid));
        prop_assert_eq!(euclid_r.to_i128(), Some(remainder));
        let (floor_q, floor_r) = a.div_rem_floor(&b).expect("unlimited nonzero divisor");
        prop_assert_eq!(floor_q.to_i128(), Some(floor));
        let expected_remainder = dividend
            .checked_sub(floor.checked_mul(divisor).expect("i64 factors fit i128"))
            .expect("remainder fits i128");
        prop_assert_eq!(floor_r.to_i128(), Some(expected_remainder));
    }
}

#[test]
fn rounded_division_handles_heap_magnitudes_and_bounded_endpoints() {
    for bits in [1, 2, 8, 64, 65, 256, 4096] {
        let width = BoundedPrecision::new(bits).expect("valid width");
        let minimum = MpInt::min_for_precision(bits);
        let negative_one = MpInt::with_precision_checked(-1_i8, width).expect("-1 fits");
        assert!(minimum.checked_div_floor(&negative_one).is_none());
        assert!(minimum.checked_div_ceil(&negative_one).is_none());
        assert!(minimum.checked_div_euclid(&negative_one).is_none());
    }
    let large = MpInt::from(1_i8) << 4096_u32;
    for a in [&large + MpInt::from(17_i8), -(&large + MpInt::from(17_i8))] {
        for b in [
            MpInt::from(3_i8),
            MpInt::from(-3_i8),
            large.clone(),
            -large.clone(),
        ] {
            let (floor, rem) = a.div_rem_floor(&b).expect("nonzero");
            assert_eq!(&floor * &b + &rem, a);
            assert_eq!(a.div_floor(&b), floor);
            let (euclid, euclid_rem) = a.div_rem_euclid(&b).expect("nonzero");
            assert_eq!(&euclid * &b + &euclid_rem, a);
            assert_eq!(a.div_euclid(&b), euclid);
            assert_eq!(a.div_ceil(&b), -(-a.clone()).div_floor(&b));
        }
    }
}
