//! Greatest-common-divisor and least-common-multiple properties.

use core::panic::AssertUnwindSafe;

use super::{std::panic::catch_unwind, *};

proptest! {
    #[test]
    fn prop_gcd_divides_both(a in strategies::uint(8), b in strategies::uint(8)) {
        let g = a.value.gcd(&b.value);
        if g.is_zero() {
            prop_assert!(a.value.is_zero() && b.value.is_zero(), "gcd=0 but inputs non-zero");
        } else {
            prop_assert_eq!(
                a.value.div_rem(&g).1,
                InternalMpUint::zero(),
                "gcd does not divide a"
            );
            prop_assert_eq!(
                b.value.div_rem(&g).1,
                InternalMpUint::zero(),
                "gcd does not divide b"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_wide_gcd_matches_euclidean_reference(
        (left_limbs, right_limbs) in (32_usize..=40).prop_flat_map(|limb_count| (
            exact_limb_vec(limb_count),
            exact_limb_vec(limb_count),
        )),
    ) {
        let left = InternalMpUint::from_limbs(left_limbs);
        let right = InternalMpUint::from_limbs(right_limbs);
        let actual = left.gcd(&right);

        let mut expected = left;
        let mut divisor = right;
        while !divisor.is_zero() {
            let remainder = expected.rem(&divisor);
            core::mem::swap(&mut expected, &mut divisor);
            divisor = remainder;
        }

        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #[test]
    fn prop_uint_bounded_lcm_matches_exact_fit(
        bits in 1_usize..=64,
        left_seed in strategies::bounded_uint_wrapped(64),
        right_seed in strategies::bounded_uint_wrapped(64),
    ) {
        let left = MpUint {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let right = MpUint {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut left_exact = left.clone();
        left_exact.precision = Precision::Unlimited;
        let mut right_exact = right.clone();
        right_exact.precision = Precision::Unlimited;
        let (exact_gcd, exact_lcm) = left_exact
            .gcd_lcm(&right_exact)
            .expect("exact gcd/lcm exists");
        let should_fit = exact_lcm.value.required_unsigned_bits_for_bounded_storage() <= bits;

        let bounded_lcm = left.lcm(&right);
        let bounded_pair = left.gcd_lcm(&right);
        prop_assert_eq!(bounded_lcm.is_some(), should_fit);
        prop_assert_eq!(bounded_pair.is_some(), should_fit);

        if let (Some(lcm), Some((gcd, paired_lcm))) = (bounded_lcm, bounded_pair) {
            prop_assert_eq!(&lcm, &exact_lcm);
            prop_assert_eq!(&gcd, &exact_gcd);
            prop_assert_eq!(&paired_lcm, &exact_lcm);
            prop_assert_eq!(lcm.precision, Precision::Bounded(nz(bits)));
            prop_assert_eq!(gcd.precision, Precision::Bounded(nz(bits)));
            prop_assert_eq!(paired_lcm.precision, Precision::Bounded(nz(bits)));

            let mut gcd_exact = gcd;
            gcd_exact.precision = Precision::Unlimited;
            let mut lcm_exact = paired_lcm;
            lcm_exact.precision = Precision::Unlimited;
            prop_assert_eq!(&gcd_exact * &lcm_exact, &left_exact * &right_exact);
        }
    }
}

proptest! {
    #[test]
    fn prop_int_bounded_lcm_matches_exact_fit(
        bits in 2_usize..=64,
        left_seed in strategies::bounded_int_wrapped(64),
        right_seed in strategies::bounded_int_wrapped(64),
    ) {
        let left = MpInt {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let right = MpInt {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut left_exact = left.clone();
        left_exact.precision = Precision::Unlimited;
        let mut right_exact = right.clone();
        right_exact.precision = Precision::Unlimited;
        let (exact_gcd, exact_lcm) = left_exact
            .gcd_lcm(&right_exact)
            .expect("exact gcd/lcm exists");
        let lcm_fits = exact_lcm.value.required_signed_bits_for_bounded_storage() <= bits;
        let pair_fits = lcm_fits
            && exact_gcd.value.required_signed_bits_for_bounded_storage() <= bits;

        let bounded_lcm = left.lcm(&right);
        let bounded_pair = left.gcd_lcm(&right);
        prop_assert_eq!(bounded_lcm.is_some(), lcm_fits);
        prop_assert_eq!(bounded_pair.is_some(), pair_fits);

        if let Some(lcm) = bounded_lcm {
            prop_assert_eq!(&lcm, &exact_lcm);
            prop_assert_eq!(lcm.precision, Precision::Bounded(nz(bits)));
        }
        if let Some((gcd, lcm)) = bounded_pair {
            prop_assert_eq!(&gcd, &exact_gcd);
            prop_assert_eq!(&lcm, &exact_lcm);
            prop_assert_eq!(gcd.precision, Precision::Bounded(nz(bits)));
            prop_assert_eq!(lcm.precision, Precision::Bounded(nz(bits)));

            let mut gcd_exact = gcd;
            gcd_exact.precision = Precision::Unlimited;
            let mut lcm_exact = lcm;
            lcm_exact.precision = Precision::Unlimited;
            let absolute_product = (&left_exact * &right_exact).abs();
            prop_assert_eq!(&gcd_exact * &lcm_exact, absolute_product);
        }

        let minimum = MpInt::min_for_precision(bits);
        let zero = MpInt::zero_with_precision(nz(bits));
        let gcd_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(minimum.gcd(&zero));
        }))
        .is_err();
        prop_assert!(
            gcd_panicked,
            "positive gcd of a signed minimum needs one more sign bit"
        );
        prop_assert!(
            minimum.gcd_lcm(&zero).is_none(),
            "positive gcd of a signed minimum needs one more sign bit"
        );
        prop_assert!(
            minimum.extended_gcd(&zero).is_none(),
            "unrepresentable gcd must reject the entire Bezout tuple"
        );
    }
}

proptest! {
    #[test]
    fn prop_gcd_commutative(a in strategies::uint(12), b in strategies::uint(12)) {
        prop_assert_eq!(a.value.gcd(&b.value), b.value.gcd(&a.value), "gcd commutative");
    }
}

proptest! {
    #[test]
    fn prop_gcd_zero(a in strategies::uint(12)) {
        let v = &a.value;
        prop_assert_eq!(v.gcd(&InternalMpUint::zero()), v.clone(), "gcd(a, 0) != a");
        prop_assert_eq!(InternalMpUint::zero().gcd(v), v.clone(), "gcd(0, a) != a");
        prop_assert_eq!(
            InternalMpUint::zero().gcd(&InternalMpUint::zero()),
            InternalMpUint::zero()
        );
    }
}

proptest! {
    #[test]
    fn prop_extended_gcd_valids(a in strategies::uint(6), b in strategies::uint(6)) {
        if !b.value.is_zero() {
            let (g, _, _) = a.value.extended_gcd(&b.value);
            prop_assert_eq!(
                a.value.div_rem(&g).1,
                InternalMpUint::zero(),
                "gcd must divide a"
            );
            prop_assert_eq!(
                b.value.div_rem(&g).1,
                InternalMpUint::zero(),
                "gcd must divide b"
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_new_public_apis_theory(val_a in strategies::uint(4), val_b in strategies::uint(4)) {
        if !val_a.is_zero() && !val_b.is_zero() {
            if let Some(l) = val_a.lcm(&val_b) {
                let g = val_a.value.gcd(&val_b.value);
                let gcd_val = MpUint { value: g, precision: val_a.precision };
                let product = &val_a * &val_b;
                let lcm_gcd_product = &l * &gcd_val;
                prop_assert_eq!(product, lcm_gcd_product, "lcm * gcd != val_a * val_b");
            }
            prop_assert_eq!(val_a.is_coprime(&val_b), val_a.value.gcd(&val_b.value).is_one(), "coprime mismatch");
        }

        if !val_a.is_zero() && !val_b.is_zero() && let Some((gcd, cof_s, _t)) = val_a.extended_gcd(&val_b) {
            let gcd_val = gcd.value;
            prop_assert_eq!(&gcd_val, &val_a.value.gcd(&val_b.value), "extended gcd mismatch");
            let term1 = &val_a * &cof_s;
            let rem = term1.value.rem(&val_b.value);
            let expected_rem = gcd_val.rem(&val_b.value);
            prop_assert_eq!(rem, expected_rem, "extended gcd cofactor check failed");
        }
    }
}

proptest! {
    #[test]
    fn prop_gcd_and_abs_diff(u1 in strategies::uint(8), u2 in strategies::uint(8), i1_val in -1000_i64..=1000_i64, i2_val in -1000_i64..=1000_i64) {
        let u1_clone = u1.clone(); let u2_clone = u2.clone();
        prop_assert_eq!(u1.gcd(&u2), u1.gcd_lcm(&u2).map_or_else(MpUint::zero, |(g, _)| g));
        let u_diff = if u1_clone >= u2_clone { &u1_clone - &u2_clone } else { &u2_clone - &u1_clone };
        prop_assert_eq!(u1.abs_diff(&u2), u_diff);

        let i1 = MpInt::from(i1_val); let i2 = MpInt::from(i2_val);
        prop_assert_eq!(i1.gcd(&i2), i1.gcd_lcm(&i2).map_or_else(MpInt::zero, |(g, _)| g));
        let abs1 = i1.abs(); let abs2 = i2.abs();
        let expected_diff = if i1.is_negative() == i2.is_negative() {
            if abs1 >= abs2 { &abs1 - &abs2 } else { &abs2 - &abs1 }
        } else { &abs1 + &abs2 };
        prop_assert_eq!(MpInt::from(i1.abs_diff(&i2)), expected_diff);
    }
}
