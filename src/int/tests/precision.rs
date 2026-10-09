//! Precision domains, numeric identity, and ambient construction policies.

#[cfg(feature = "std")]
use alloc::string::ToString;

use proptest::prelude::{Just, any, prop_assert, prop_assert_eq, prop_oneof, proptest};

#[cfg(feature = "std")]
use crate::PrecisionContext;
use crate::{AmbientPrecision, BoundedPrecision, MpInt, MpUint, Precision};

#[cfg(feature = "std")]
use super::support::hash_u64;
use super::{strategies, support::nz};

proptest! {
    #[test]
    fn bounded_precision_has_one_canonical_domain(bits in prop_oneof![Just(0), Just(usize::MAX), any::<usize>()]) {
        let width_result = BoundedPrecision::new(bits);
        let valid = (1..usize::MAX).contains(&bits);
        prop_assert_eq!(width_result.is_some(), valid);
        prop_assert_eq!(Precision::new_bounded(bits).is_some(), valid);
        prop_assert_eq!(AmbientPrecision::new_bounded(bits).is_some(), valid);
        if let Some(width) = width_result {
            prop_assert_eq!(width.get(), bits);
            prop_assert_eq!(Precision::new_bounded(bits), Some(Precision::Bounded(width)));
            prop_assert_eq!(AmbientPrecision::new_bounded(bits), Some(AmbientPrecision::Bounded(width)));
            prop_assert_eq!(Precision::from(AmbientPrecision::Bounded(width)), Precision::Bounded(width));
            prop_assert_eq!(Precision::Bounded(width).significant_bits(), Some(bits));
        }
        prop_assert_eq!(Precision::from(AmbientPrecision::Unset), Precision::Unlimited);
        prop_assert_eq!(Precision::from(AmbientPrecision::Unlimited), Precision::Unlimited);
        prop_assert_eq!(Precision::Unlimited.significant_bits(), None);
    }

    #[test]
    fn numeric_identity_and_order_ignore_precision(
        unsigned in strategies::uint(16), other in strategies::uint(16),
        signed in strategies::int(16), extra in 0_usize..=64,
    ) {
        let unsigned_bits = unsigned.significant_bits().max(1) + extra;
        let bounded_unsigned = MpUint::with_precision_checked(unsigned.clone(), nz(unsigned_bits)).expect("value fits");
        let signed_bits = signed.significant_bits() + 1 + extra;
        let bounded_signed = MpInt::with_precision_checked(signed.clone(), nz(signed_bits)).expect("value fits");
        prop_assert_eq!(&unsigned, &bounded_unsigned);
        prop_assert_eq!(&signed, &bounded_signed);
        prop_assert_eq!(unsigned.cmp(&other), bounded_unsigned.cmp(&other));
        prop_assert_eq!(unsigned.cmp(&bounded_unsigned), core::cmp::Ordering::Equal);
        prop_assert_eq!(signed.cmp(&bounded_signed), core::cmp::Ordering::Equal);
        #[cfg(feature = "std")]
        {
            prop_assert_eq!(hash_u64(&unsigned), hash_u64(&bounded_unsigned));
            prop_assert_eq!(hash_u64(&signed), hash_u64(&bounded_signed));
            if unsigned == other { prop_assert_eq!(hash_u64(&unsigned), hash_u64(&other)); }
        }
        prop_assert_eq!(&MpInt::from(bounded_unsigned), &unsigned);
    }

    #[test]
    fn binary_operations_combine_widths_and_assignments_preserve_destination(
        left_seed in any::<u64>(), right_seed in any::<u64>(),
        left_bits in 1_usize..=64, right_bits in 1_usize..=64,
    ) {
        let left = MpUint::with_precision_wrapping(left_seed, nz(left_bits));
        let right = MpUint::with_precision_wrapping(right_seed, nz(right_bits));
        let combined = Precision::Bounded(nz(left_bits.max(right_bits)));
        prop_assert_eq!(left.wrapping_add(&right).precision(), combined);
        prop_assert_eq!(left.wrapping_mul(&right).precision(), combined);
        prop_assert_eq!((&left & &right).precision(), combined);
        let exact_left = MpUint::zero() + &left;
        let exact_right = MpUint::zero() + &right;
        let sum = &exact_left + &exact_right;
        prop_assert_eq!((&left + &exact_right).precision(), Precision::Unlimited);
        for (a, b) in [(&left, &exact_right), (&exact_left, &right)] {
            for (actual, exact) in [(a.overflowing_add(b), &exact_left + &exact_right), (a.overflowing_mul(b), &exact_left * &exact_right)] {
                prop_assert_eq!(&actual.0, &exact);
                prop_assert_eq!(actual.0.precision(), Precision::Unlimited);
                prop_assert!(!actual.1);
            }
        }
        let signed_left = MpInt::with_precision_wrapping(left_seed, nz(left_bits));
        let signed_right = MpInt::with_precision_wrapping(right_seed, nz(right_bits));
        let exact_signed_left = MpInt::zero() + &signed_left;
        let exact_signed_right = MpInt::zero() + &signed_right;
        for (a, b) in [(&signed_left, &exact_signed_right), (&exact_signed_left, &signed_right)] {
            for (actual, exact) in [(a.overflowing_add(b), &exact_signed_left + &exact_signed_right), (a.overflowing_mul(b), &exact_signed_left * &exact_signed_right)] {
                prop_assert_eq!(&actual.0, &exact);
                prop_assert_eq!(actual.0.precision(), Precision::Unlimited);
                prop_assert!(!actual.1);
            }
        }
        if sum.significant_bits() <= left_bits {
            for owned in [false, true] {
                let mut destination = left.clone();
                if owned { destination += exact_right.clone(); } else { destination += &exact_right; }
                prop_assert_eq!(&destination, &sum);
                prop_assert_eq!(destination.precision(), left.precision());
            }
        }
        prop_assert_eq!(exact_left.checked_add(&exact_right), Some(sum));
        prop_assert_eq!(exact_left.checked_mul(&exact_right), Some(&exact_left * &exact_right));
        if exact_left >= exact_right {
            prop_assert_eq!(exact_left.checked_sub(&exact_right), Some(&exact_left - &exact_right));
        }
    }

    #[test]
    fn constructors_without_ambient_context_are_unlimited(value in any::<u64>(), small in any::<u8>(), medium in any::<u16>()) {
        prop_assert_eq!(MpUint::from(value).precision(), Precision::Unlimited);
        prop_assert_eq!(MpUint::from(small).precision(), Precision::Unlimited);
        prop_assert_eq!(MpUint::from(medium).precision(), Precision::Unlimited);
        prop_assert_eq!(MpUint::default().precision(), Precision::Unlimited);
        prop_assert_eq!(MpInt::default().precision(), Precision::Unlimited);
    }
}

#[cfg(feature = "std")]
proptest! {
    #[test]
    fn nested_contexts_preserve_full_widths_and_existing_values(
        bits in prop_oneof![Just(1_usize), Just(usize::MAX - 1), 1_usize..usize::MAX],
        nested_bits in prop_oneof![Just(1_usize), Just(usize::MAX - 1), 1_usize..usize::MAX],
        left in any::<u64>(), right in any::<u64>(),
    ) {
        let initial = PrecisionContext::active();
        let a = MpUint::zero() + MpUint::from(left);
        let b = MpUint::zero() + MpUint::from(right);
        PrecisionContext::with_bounded(bits, || {
            let outer = AmbientPrecision::Bounded(nz(bits));
            prop_assert_eq!(PrecisionContext::active(), outer);
            prop_assert_eq!(MpUint::with_capacity(4).precision(), Precision::Bounded(nz(bits)));
            prop_assert_eq!(MpInt::with_capacity(4).precision(), Precision::Bounded(nz(bits)));
            PrecisionContext::with_unlimited(|| {
                prop_assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                PrecisionContext::with_bounded(nested_bits, || {
                    prop_assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(nz(nested_bits)));
                    Ok(())
                })?;
                prop_assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                Ok(())
            })?;
            prop_assert_eq!(PrecisionContext::active(), outer);
            let sum = &a + &b;
            prop_assert_eq!(sum.precision(), Precision::Unlimited);
            prop_assert_eq!(sum.to_u128(), Some(u128::from(left) + u128::from(right)));
            Ok(())
        })?;
        prop_assert_eq!(PrecisionContext::active(), initial);
    }

    #[test]
    fn ambient_constructors_widen_and_parsers_enforce_width(
        bits in 1_usize..=128, unsigned in any::<u128>(), signed in any::<i128>(),
    ) {
        let unsigned_required = (128 - unsigned.leading_zeros()).max(1) as usize;
        let signed_required = (129 - (if signed < 0 { !signed } else { signed }).cast_unsigned().leading_zeros()) as usize;
        PrecisionContext::with_bounded(bits, || {
            let u = MpUint::from(unsigned);
            let i = MpInt::from(signed);
            prop_assert_eq!(u.precision(), Precision::Bounded(nz(bits.max(unsigned_required))));
            prop_assert_eq!(i.precision(), Precision::Bounded(nz(bits.max(signed_required))));
            prop_assert_eq!(MpUint::default().precision(), Precision::Unlimited);
            prop_assert_eq!(MpInt::default().precision(), Precision::Unlimited);
            for parsed in [unsigned.to_string().parse::<MpUint>(), MpUint::from_str_radix(&unsigned.to_string(), 10)] {
                prop_assert_eq!(parsed.is_ok(), unsigned_required <= bits);
                if let Ok(value) = parsed {
                    prop_assert_eq!(value.precision(), Precision::Bounded(nz(bits)));
                    prop_assert_eq!(value.to_u128(), Some(unsigned));
                }
            }
            for parsed in [signed.to_string().parse::<MpInt>(), MpInt::from_str_radix(&signed.to_string(), 10)] {
                prop_assert_eq!(parsed.is_ok(), signed_required <= bits);
                if let Ok(value) = parsed {
                    prop_assert_eq!(value.precision(), Precision::Bounded(nz(bits)));
                    prop_assert_eq!(value.to_i128(), Some(signed));
                }
            }
            Ok(())
        })?;
    }
}

#[cfg(feature = "std")]
#[test]
#[expect(
    clippy::panic,
    reason = "Injected unwinding verifies restoration of nested precision scopes"
)]
fn precision_scopes_restore_after_panics_and_remain_thread_local() {
    let initial = PrecisionContext::active();
    PrecisionContext::with_bounded(73, || {
        let outer = AmbientPrecision::Bounded(nz(73));
        let panic = std::panic::catch_unwind(|| {
            PrecisionContext::with_unlimited(|| {
                assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                PrecisionContext::with_bounded(129, || {
                    assert_eq!(
                        PrecisionContext::active(),
                        AmbientPrecision::Bounded(nz(129))
                    );
                    panic!("exercise nested precision restoration");
                });
            });
        });
        assert!(panic.is_err());
        assert_eq!(PrecisionContext::active(), outer);
        for invalid in [0, usize::MAX] {
            assert!(
                std::panic::catch_unwind(|| PrecisionContext::with_bounded(invalid, || {}))
                    .is_err()
            );
            assert_eq!(PrecisionContext::active(), outer);
        }
        for required in [0_usize, 1, 72, 73, 74] {
            let value = if required == 0 {
                0
            } else {
                1_u128 << (required - 1)
            };
            assert_eq!(
                MpUint::from(value).precision(),
                Precision::Bounded(nz(required.max(73)))
            );
        }
    });
    PrecisionContext::with_bounded(usize::MAX - 1, || {
        let outer = AmbientPrecision::Bounded(nz(usize::MAX - 1));
        let child = std::thread::spawn(move || {
            assert_eq!(PrecisionContext::active(), initial);
            PrecisionContext::with_bounded(1, || {
                assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(nz(1)));
                PrecisionContext::with_unlimited(|| {
                    assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                });
                assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(nz(1)));
            });
            assert_eq!(PrecisionContext::active(), initial);
        });
        child.join().expect("child precision checks succeed");
        assert_eq!(PrecisionContext::active(), outer);
    });
    assert_eq!(PrecisionContext::active(), initial);
    PrecisionContext::with_bounded(8, || {
        for (text, fits) in [("255", true), ("256", false)] {
            assert_eq!(text.parse::<MpUint>().is_ok(), fits);
            assert_eq!(MpUint::from_str_radix(text, 10).is_ok(), fits);
        }
        for (text, fits) in [("-128", true), ("128", false)] {
            assert_eq!(text.parse::<MpInt>().is_ok(), fits);
            assert_eq!(MpInt::from_str_radix(text, 10).is_ok(), fits);
        }
    });
}
