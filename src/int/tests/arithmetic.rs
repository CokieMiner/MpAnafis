//! Algebraic identities and division reconstruction through the public API.

use proptest::prelude::{ProptestConfig, any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, Precision};

use super::{strategies, support::nz};

proptest! {
    #[test]
    fn unsigned_arithmetic_obeys_ring_and_division_identities(
        a in strategies::uint(16), b in strategies::uint(16), c in strategies::uint(16),
    ) {
        prop_assert_eq!(&a + &b, &b + &a);
        prop_assert_eq!(&(&a + &b) + &c, &a + &(&b + &c));
        prop_assert_eq!(&(&a + MpUint::zero()), &a);
        prop_assert_eq!(&(MpUint::zero() + &a), &a);
        prop_assert_eq!(&(&(&a + &b) - &b), &a);
        prop_assert_eq!(&a * &b, &b * &a);
        prop_assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
        prop_assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
        prop_assert_eq!(&(&a * MpUint::one()), &a);
        prop_assert_eq!(&a * MpUint::zero(), MpUint::zero());
        prop_assert_eq!(MpUint::zero() * &a, MpUint::zero());
        prop_assert_eq!(&(&a / MpUint::one()), &a);
        prop_assert_eq!(a.mul_add(&b, &c), &a * &b + &c);
        prop_assert_eq!(a.midpoint(&b), (&a + &b) >> 1_usize);
        if !b.is_zero() {
            let q = &a / &b;
            let r = &a % &b;
            prop_assert!(r < b);
            prop_assert_eq!(&(&q * &b + &r), &a);
            prop_assert_eq!(a.checked_div(&b), Some(q.clone()));
            prop_assert_eq!(a.checked_rem(&b), Some(r.clone()));
            prop_assert_eq!(a.is_divisible_by(&b), r.is_zero());
            prop_assert_eq!(b.is_divisor_of(&a), r.is_zero());
            for actual in [a.div_trunc(&b), a.div_euclid(&b), a.div_floor(&b)] {
                prop_assert_eq!(&actual, &q);
            }
            for actual in [a.rem_trunc(&b), a.rem_euclid(&b), a.mod_floor(&b)] {
                prop_assert_eq!(&actual, &r);
            }
            prop_assert_eq!(a.div_ceil(&b), if r.is_zero() { q } else { q + MpUint::one() });
        }
        if !a.is_zero() { prop_assert_eq!(&a / &a, MpUint::one()); }
    }

    #[test]
    fn signed_arithmetic_obeys_sign_and_division_identities(
        a in strategies::int(32), b in strategies::int(32), c in strategies::int(16),
    ) {
        prop_assert_eq!(&(-&(-&a)), &a);
        prop_assert_eq!(&a + &(-&b), &a - &b);
        prop_assert!(!a.abs().is_negative());
        let product = &a * &b;
        prop_assert_eq!(product.is_negative(), !a.is_zero() && !b.is_zero() && a.is_negative() != b.is_negative());
        prop_assert_eq!(a.mul_add(&b, &c), &a * &b + &c);
        prop_assert_eq!(a.midpoint(&b), (&a + &b) / MpInt::from(2_u8));
        prop_assert_eq!(&(&a / MpInt::one()), &a);
        prop_assert_eq!(&a / MpInt::minus_one(), -&a);
        let paired = a.div_rem(&b);
        prop_assert_eq!(paired.is_none(), a.checked_div(&b).is_none());
        if let Some((q, r)) = paired {
            prop_assert_eq!(a.checked_div(&b), Some(q.clone()));
            prop_assert_eq!(a.checked_rem(&b), Some(r.clone()));
            prop_assert_eq!(&(&q * &b + &r), &a);
            prop_assert_eq!(a.is_divisible_by(&b), r.is_zero());
            prop_assert_eq!(b.is_divisor_of(&a), r.is_zero());
            prop_assert!(r.abs() < b.abs());
            if !r.is_zero() { prop_assert_eq!(r.is_negative(), a.is_negative()); }
            let (euclid_q, euclid_r) = a.div_rem_euclid(&b).expect("nonzero divisor");
            prop_assert!(euclid_r >= MpInt::zero() && euclid_r < b.abs());
            prop_assert_eq!(&(&b * &euclid_q + &euclid_r), &a);
            prop_assert_eq!(a.div_euclid(&b), euclid_q);
            prop_assert_eq!(a.rem_euclid(&b), euclid_r);
            let (floor_q, floor_r) = a.div_rem_floor(&b).expect("nonzero divisor");
            if !floor_r.is_zero() { prop_assert_eq!(floor_r.is_negative(), b.is_negative()); }
            prop_assert_eq!(&(&b * &floor_q + &floor_r), &a);
            prop_assert_eq!(a.div_floor(&b), floor_q);
            prop_assert_eq!(a.mod_floor(&b), floor_r);
        } else { prop_assert!(b.is_zero()); }
        let zero = &a - &a;
        prop_assert!(zero.is_zero() && !zero.is_negative() && !(-zero).is_negative());
    }

    #[test]
    fn bounded_unsigned_division_preserves_combined_and_destination_precision(
        lhs_bits in 1_usize..=128, rhs_bits in 1_usize..=128,
        lhs_seed in any::<u128>(), rhs_seed in any::<u128>(),
    ) {
        let lhs = MpUint::with_precision_wrapping(lhs_seed, nz(lhs_bits));
        let rhs = MpUint::with_precision_wrapping(rhs_seed | 1, nz(rhs_bits));
        let combined = Precision::Bounded(nz(lhs_bits.max(rhs_bits)));
        let q = &lhs / &rhs;
        let r = &lhs % &rhs;
        prop_assert_eq!(q.precision(), combined);
        prop_assert_eq!(r.precision(), combined);
        prop_assert_eq!(&((MpUint::zero() + &q) * &rhs + &r), &lhs);
        prop_assert!(r < rhs);
        let mut quotient = lhs.clone();
        quotient /= &rhs;
        prop_assert_eq!(&quotient, &q);
        prop_assert_eq!(quotient.precision(), lhs.precision());
        let mut remainder = lhs.clone();
        remainder %= &rhs;
        prop_assert_eq!(&remainder, &r);
        prop_assert_eq!(remainder.precision(), lhs.precision());
        let expected = if r.is_zero() { MpUint::zero() + q } else { MpUint::zero() + q + MpUint::one() };
        let ceiling = lhs.div_ceil(&rhs);
        prop_assert_eq!(&ceiling, &expected);
        prop_assert_eq!(ceiling.precision(), combined);
        let checked = lhs.checked_div_ceil(&rhs).expect("nonzero divisor");
        prop_assert_eq!(&checked, &expected);
        prop_assert_eq!(checked.precision(), combined);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    #[cfg_attr(miri, ignore = "Division and reconstruction of 625-limb values exceed the interpreter test budget")]
    fn wide_unsigned_division_reconstructs_the_dividend(a in strategies::uint(625), b in strategies::uint_nonzero(625)) {
        let (q, r) = a.div_rem(&b).expect("nonzero divisor");
        prop_assert!(r < b);
        prop_assert_eq!(&q * &b + &r, a);
    }
}
