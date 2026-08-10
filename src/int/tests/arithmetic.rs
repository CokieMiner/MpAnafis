//! Algebraic and signed arithmetic properties.

use super::*;

proptest! {
    #[test]
    fn prop_add_commutative(a in strategies::uint(16), b in strategies::uint(16)) {
        let ab = &a + &b;
        let ba = &b + &a;
        prop_assert_eq!(&ab, &ba, "add commutative");
        #[cfg(feature = "std")]
        prop_assert_eq!(hash_u64(&ab), hash_u64(&ba), "add commutative hash");
    }
}

proptest! {
    #[test]
    fn prop_int_div_rem_invariants(
        a in strategies::int(32),
        b in strategies::int(32),
    ) {
        let checked_div = a.checked_div(&b);
        let checked_rem = a.checked_rem(&b);
        let div_rem = a.div_rem(&b);

        prop_assert_eq!(div_rem.is_none(), checked_div.is_none());

        if let Some((q, r)) = div_rem {
            prop_assert_eq!(q, checked_div.expect("checked_div should succeed"));
            prop_assert_eq!(r, checked_rem.expect("checked_rem should succeed"));
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_div_rem_wide_identity(
        a in strategies::uint(625),
        b in strategies::uint_nonzero(625),
    ) {
        let q = &a / &b;
        let r = &a % &b;
        prop_assert!(r < b, "remainder not reduced");
        prop_assert_eq!(&q * &b + &r, a);
    }
}

proptest! {
    #[test]
    fn prop_uint_math_api_invariants(
        ua in strategies::uint(16),
        ub in strategies::uint(16),
        uc in strategies::uint(16),
    ) {
        let u_mul_add = ua.mul_add(&ub, &uc);
        prop_assert_eq!(u_mul_add, (&ua * &ub) + &uc, "Uint mul_add");

        let u_mid = ua.midpoint(&ub);
        prop_assert_eq!(u_mid, (&ua + &ub) >> 1_usize, "Uint midpoint");

        if !ub.is_zero() {
            let rem = &ua % &ub;
            prop_assert_eq!(ua.is_divisible_by(&ub), rem.is_zero());
            prop_assert_eq!(ub.is_divisor_of(&ua), rem.is_zero());

            prop_assert_eq!(ua.div_trunc(&ub), &ua / &ub);
            prop_assert_eq!(ua.rem_trunc(&ub), &ua % &ub);
            prop_assert_eq!(ua.div_euclid(&ub), &ua / &ub);
            prop_assert_eq!(ua.rem_euclid(&ub), &ua % &ub);
            prop_assert_eq!(ua.div_floor(&ub), &ua / &ub);
            prop_assert_eq!(ua.mod_floor(&ub), &ua % &ub);

            let ceil_q = ua.div_ceil(&ub);
            let expected_ceil = if rem.is_zero() {
                &ua / &ub
            } else {
                (&ua / &ub) + ArbiUint::one()
            };
            prop_assert_eq!(ceil_q, expected_ceil, "Uint div_ceil");
        }
    }
}

#[test]
fn bounded_ceiling_division_preserves_combined_precision() {
    let wide = nz(8);
    let narrow = nz(4);
    let unsigned_dividend =
        ArbiUint::with_precision_checked(7_u8, wide).expect("seven fits eight bits");
    let unsigned_divisor =
        ArbiUint::with_precision_checked(3_u8, narrow).expect("three fits four bits");
    let expected_precision = Precision::Bounded(wide);

    let unsigned = unsigned_dividend.div_ceil(&unsigned_divisor);
    let unsigned_checked = unsigned_dividend
        .checked_div_ceil(&unsigned_divisor)
        .expect("the divisor is non-zero");
    assert_eq!(unsigned.value, InternalArbiUint::from_limb(3));
    assert_eq!(unsigned.precision, expected_precision);
    assert_eq!(unsigned_checked.precision, expected_precision);

    let signed_dividend =
        ArbiInt::with_precision_checked(7_i8, wide).expect("seven fits eight signed bits");
    let signed_divisor =
        ArbiInt::with_precision_checked(3_i8, narrow).expect("three fits four signed bits");
    let signed_checked = signed_dividend
        .checked_div_ceil(&signed_divisor)
        .expect("the divisor is non-zero and the quotient fits");
    assert_eq!(signed_checked, ArbiInt::from(3_i8));
    assert_eq!(signed_checked.precision, expected_precision);
}

proptest! {
    #[test]
    fn prop_bounded_uint_division_results_fit_without_post_checks(
        lhs_bits in 1_usize..=128,
        rhs_bits in 1_usize..=128,
        lhs_seed in any::<u128>(),
        rhs_seed in any::<u128>(),
    ) {
        let lhs_width = nz(lhs_bits);
        let rhs_width = nz(rhs_bits);
        let lhs = ArbiUint::with_precision_wrapping(lhs_seed, lhs_width);
        // Setting the low bit before wrapping keeps the divisor non-zero for
        // every generated width, including one bit.
        let rhs = ArbiUint::with_precision_wrapping(rhs_seed | 1, rhs_width);
        let combined_precision = Precision::Bounded(nz(lhs_bits.max(rhs_bits)));

        let quotient = &lhs / &rhs;
        let remainder = &lhs % &rhs;
        prop_assert_eq!(quotient.precision, combined_precision);
        prop_assert_eq!(remainder.precision, combined_precision);
        let reconstructed = quotient.value.mul(&rhs.value).add(&remainder.value);
        prop_assert_eq!(
            &reconstructed,
            &lhs.value,
            "q * rhs + r must reconstruct lhs"
        );
        prop_assert!(remainder.value < rhs.value);

        let mut quotient_assign = lhs.clone();
        quotient_assign /= &rhs;
        prop_assert_eq!(quotient_assign.precision, Precision::Bounded(lhs_width));
        prop_assert_eq!(&quotient_assign.value, &quotient.value);

        let mut remainder_assign = lhs.clone();
        remainder_assign %= &rhs;
        prop_assert_eq!(remainder_assign.precision, Precision::Bounded(lhs_width));
        prop_assert_eq!(&remainder_assign.value, &remainder.value);

        let ceiling = lhs.div_ceil(&rhs);
        let checked_ceiling = lhs
            .checked_div_ceil(&rhs)
            .expect("a generated divisor is non-zero");
        let mut expected_ceiling = quotient.value;
        if !remainder.value.is_zero() {
            expected_ceiling.increment();
        }
        prop_assert_eq!(&ceiling.value, &expected_ceiling);
        prop_assert_eq!(&checked_ceiling.value, &expected_ceiling);
        prop_assert_eq!(ceiling.precision, combined_precision);
        prop_assert_eq!(checked_ceiling.precision, combined_precision);
    }
}

proptest! {
    #[test]
    fn prop_int_math_api_invariants(
        ia in strategies::int(16),
        ib in strategies::int(16),
        ic in strategies::int(16),
    ) {
        let i_mul_add = ia.mul_add(&ib, &ic);
        prop_assert_eq!(i_mul_add, (&ia * &ib) + &ic, "Int mul_add");

        let i_mid = ia.midpoint(&ib);
        prop_assert_eq!(i_mid, (&ia + &ib) >> 1_usize, "Int midpoint");

        if !ib.is_zero() {
            let rem = &ia % &ib;
            prop_assert_eq!(ia.is_divisible_by(&ib), rem.is_zero());
            prop_assert_eq!(ib.is_divisor_of(&ia), rem.is_zero());

            let (euclid_q, euclid_r) = ia.div_rem_euclid(&ib).expect("non-zero div");
            prop_assert!(euclid_r >= ArbiInt::zero(), "euclid_r must be non-negative");
            prop_assert!(euclid_r < ib.abs(), "euclid_r must be less than |b|");
            let euclid_recon = (&ib * &euclid_q) + &euclid_r;
            prop_assert_eq!(&euclid_recon, &ia, "Euclid reconstruction");
            prop_assert_eq!(ia.div_euclid(&ib), euclid_q);
            prop_assert_eq!(ia.rem_euclid(&ib), euclid_r);

            let (floor_q, floor_r) = ia.div_rem_floor(&ib).expect("non-zero div");
            if !floor_r.is_zero() {
                prop_assert_eq!(floor_r.is_negative(), ib.is_negative(), "floor_r sign must match b sign");
            }
            let floor_recon = (&ib * &floor_q) + &floor_r;
            prop_assert_eq!(&floor_recon, &ia, "Floor reconstruction");
            prop_assert_eq!(ia.div_floor(&ib), floor_q);
            prop_assert_eq!(ia.mod_floor(&ib), floor_r);
        }
    }
}

proptest! {
    #[test]
    fn prop_add_associative(a in strategies::uint(12), b in strategies::uint(12), c in strategies::uint(12)) {
        prop_assert_eq!(&(&a + &b) + &c, &a + &(&b + &c), "add associative");
    }
}

proptest! {
    #[test]
    fn prop_add_identity(a in strategies::uint(16)) {
        let zero = ArbiUint::zero();
        let a_plus_0 = &a + &zero;
        prop_assert!(a_plus_0 == a, "a + 0 != a");
        prop_assert!(&zero + &a == a, "0 + a != a");
    }
}

proptest! {
    #[test]
    fn prop_add_sub_inverse(a in strategies::uint(12), b in strategies::uint(12)) {
        if a >= b {
            let sum = &a + &b;
            prop_assert_eq!(&sum - &b, a, "(a + b) - b != a");
        }
    }
}

proptest! {
    #[test]
    fn prop_mul_commutative(a in strategies::uint(12), b in strategies::uint(12)) {
        let ab = &a * &b;
        let ba = &b * &a;
        prop_assert_eq!(&ab, &ba, "mul commutative");
        #[cfg(feature = "std")]
        prop_assert_eq!(hash_u64(&ab), hash_u64(&ba), "mul commutative hash");
    }
}

proptest! {
    #[test]
    fn prop_mul_associative(a in strategies::uint(8), b in strategies::uint(8), c in strategies::uint(8)) {
        prop_assert_eq!(&(&a * &b) * &c, &a * &(&b * &c), "mul associative");
    }
}

proptest! {
    #[test]
    fn prop_mul_identity(a in strategies::uint(12)) {
        let one = ArbiUint::one();
        let zero = ArbiUint::zero();
        prop_assert!(&a * &one == a, "a * 1 != a");
        prop_assert_eq!(&a * &zero, ArbiUint::zero(), "a * 0 != 0");
        prop_assert_eq!(&zero * &a, ArbiUint::zero(), "0 * a != 0");
    }
}

proptest! {
    #[test]
    fn prop_mul_distributive(a in strategies::uint(6), b in strategies::uint(6), c in strategies::uint(6)) {
        prop_assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c), "distributive");
    }
}

proptest! {
    #[test]
    fn prop_div_rem_identity(a in strategies::uint(12), b in strategies::uint_nonzero(12)) {
        let q = &a / &b;
        let r = &a % &b;
        prop_assert!(r < b, "remainder >= divisor");
        prop_assert_eq!(&(&q * &b) + &r, a, "a = q*b + r failed");
    }
}

proptest! {
    #[test]
    fn prop_div_self_is_one(a in strategies::uint_nonzero(12)) {
        prop_assert_eq!(&a / &a, ArbiUint::one(), "a/a != 1");
    }
}

proptest! {
    #[test]
    fn prop_div_one_is_self(a in strategies::uint(12)) {
        let one = ArbiUint::one();
        prop_assert_eq!(&a / &one, a, "a/1 != a");
    }
}

proptest! {
    #[test]
    fn prop_signed_neg_involutive(a in strategies::int(12)) {
        prop_assert_eq!(-&(-&a), a, "signed: -(-a) != a");
    }
}

proptest! {
    #[test]
    fn prop_signed_add_neg_is_sub(a in strategies::int(8), b in strategies::int(8)) {
        prop_assert_eq!(&a + &(-&b), &a - &b, "a + (-b) != a - b");
    }
}

proptest! {
    #[test]
    fn prop_signed_mul_sign_rules(a in strategies::int(8), b in strategies::int(8)) {
        let prod = &a * &b;
        let expected_positive = (a.is_positive() == b.is_positive()) || a.value.abs.is_zero() || b.value.abs.is_zero();
        prop_assert_eq!(prod.is_positive() || prod.value.abs.is_zero(), expected_positive, "sign of product wrong");
    }
}

proptest! {
    #[test]
    fn prop_signed_abs_non_negative(a in strategies::int(12)) {
        let abs_a = a.abs();
        prop_assert!(abs_a.is_positive() || abs_a.value.abs.is_zero(), "abs of signed should be non-negative");
    }
}

proptest! {
    #[test]
    fn prop_no_negative_zero_signed(a in strategies::int(12)) {
        let neg_a = -&a;
        if a.value.abs.is_zero() {
            prop_assert!(neg_a.value.is_positive, "neg of zero must have is_positive=true");
            prop_assert!(neg_a.value.abs.is_zero(), "neg of zero must have zero magnitude");
        }
    }
}

proptest! {
    #[test]
    fn prop_no_negative_zero_constructors(value in -1000_i32..=1000_i32) {
        let integer = ArbiInt::from(value);
        if integer.value.abs.is_zero() {
            prop_assert!(
                integer.value.is_positive,
                "ArbiInt from {value}: zero must have is_positive=true"
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_div_trunc_sign_matches_rust(a in -1000_i64..=1000_i64, b in strategies::int_nonzero(4)) {
        let ai = ArbiInt::from(a);
        let bi = b;
        let q = &ai / &bi;
        let r = &ai % &bi;
        prop_assert_eq!(&(&q * &bi) + &r, ai, "a = q*b + r failed");
    }
}

proptest! {
    #[test]
    fn prop_div_trunc_sign_cases_rust_equivalent(a in -1000_i64..=1000_i64, b in (-1000_i64..=1000_i64).prop_filter("non-zero", |v| *v != 0)) {
        let ai = ArbiInt::from(a);
        let bi = ArbiInt::from(b);
        let q = &ai / &bi;
        let r = &ai % &bi;
        prop_assert_eq!(&(&q * &bi) + &r, ai, "a = q*b + r failed");
        let (expected_q, expected_r) = (a.checked_div(b).expect("non-zero"), a.checked_rem(b).expect("non-zero"));
        if a.checked_div(b) == Some(expected_q) && a.checked_rem(b) == Some(expected_r) {
            let _ = (a, b);
            prop_assert_eq!(q, ArbiInt::from(expected_q), "{} / {}", a, b);
            prop_assert_eq!(r, ArbiInt::from(expected_r), "{} % {}", a, b);
        }
    }
}

proptest! {
    #[test]
    fn prop_div_by_one_min(n in any::<i64>()) {
        let a = ArbiInt::from(n);
        let a_clone = a.clone();
        let one = ArbiInt::from(1_i8);
        let neg_one = ArbiInt::from(-1_i8);
        prop_assert_eq!(&a / &one, a_clone, "{} / 1", n);
        if n == i64::MIN {
            // i64::MIN / -1 = 2^63 (i64::MAX + 1)
            let expected = ArbiInt::from(i64::MAX) + ArbiInt::from(1_i8);
            prop_assert_eq!(&a / &neg_one, expected, "i64::MIN / -1");
        } else {
            // a / -1 == -a (safe for all i64 values except i64::MIN)
            prop_assert_eq!(&a / &neg_one, -&a, "{} / -1", n);
        }
    }
}

#[cfg(feature = "std")]
proptest! {
    #[test]
    #[should_panic(expected = "division overflow")]
    fn prop_bounded_min_div_neg_one_panics(bits in 2_usize..=127) {
        let shift = u32::try_from(bits.wrapping_sub(1)).expect("width fits u32");
        let minimum = 1_i128
            .checked_shl(shift)
            .expect("property width is at most 127")
            .wrapping_neg();
        let min_value = ArbiInt::with_precision_checked(minimum, nz(bits))
            .expect("signed minimum fits its width");
        let neg_one = ArbiInt::with_precision_checked(-1_i8, nz(bits))
            .expect("negative one fits every signed width");
        drop(min_value / neg_one);
    }
}

proptest! {
    #[test]
    fn prop_ops_traits_unlimited(a in strategies::int_maybe_bounded(8), b in strategies::int_maybe_bounded(8)) {
        let mut a_mut = a; let mut b_mut = b;
        a_mut.precision = Precision::Unlimited;
        b_mut.precision = Precision::Unlimited;

        prop_assert_eq!(&a_mut + &b_mut, a_mut.checked_add(&b_mut).expect("add"));
        let mut a_add = a_mut.clone(); a_add += &b_mut;
        prop_assert_eq!(a_add, a_mut.checked_add(&b_mut).expect("add"));

        prop_assert_eq!(&a_mut - &b_mut, a_mut.checked_sub(&b_mut).expect("sub"));
        let mut a_sub = a_mut.clone(); a_sub -= &b_mut;
        prop_assert_eq!(a_sub, a_mut.checked_sub(&b_mut).expect("sub"));

        prop_assert_eq!(&a_mut * &b_mut, a_mut.checked_mul(&b_mut).expect("mul"));
        let mut a_mul = a_mut.clone(); a_mul *= &b_mut;
        prop_assert_eq!(a_mul, a_mut.checked_mul(&b_mut).expect("mul"));

        if !b_mut.is_zero() {
            prop_assert_eq!(&a_mut / &b_mut, a_mut.checked_div(&b_mut).expect("div"));
            prop_assert_eq!(&a_mut % &b_mut, a_mut.checked_rem(&b_mut).expect("rem"));
            let mut a_div = a_mut.clone(); a_div /= &b_mut;
            prop_assert_eq!(a_div, a_mut.checked_div(&b_mut).expect("div"));
            let mut a_rem = a_mut.clone(); a_rem %= &b_mut;
            prop_assert_eq!(a_rem, a_mut.checked_rem(&b_mut).expect("rem"));
        }
    }
}
