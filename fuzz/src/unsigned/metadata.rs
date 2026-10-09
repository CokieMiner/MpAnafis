//! Constructors, storage operations, and destination-precision assignment.

use mp_anafis::{AmbientPrecision, BoundedPrecision, MpError, MpUint, Precision, PrecisionContext};
use rug::{Integer, integer::Order};

use crate::{Bounds, Input, assert_integer};

pub fn fuzz_all(a: &MpUint, ra: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let precision = BoundedPrecision::new(bits).unwrap();
    let bounds = Bounds {
        bits: Some(bits),
        signed: false,
    };
    match input.operation % 4 {
        0 => {
            let mut value = MpUint::new(a.clone());
            let metadata = value.precision();
            let limbs = usize::try_from(ra.significant_bits())
                .unwrap()
                .div_ceil(usize::BITS as usize);
            assert!(value.capacity() >= limbs);
            value.reserve(usize::from(input.parameter % 64));
            value.reserve_exact(usize::from(input.parameter % 64));
            value.shrink_to_fit();
            assert!(value.capacity() >= limbs);
            assert_integer(&value, ra);
            assert_eq!(value.precision(), metadata);
            let mut other = MpUint::zero_with_precision(precision);
            value.swap(&mut other);
            assert_eq!(value.precision(), Precision::Bounded(precision));
            assert!(value.is_zero());
            assert_integer(&other, ra);
            assert_eq!(other.precision(), metadata);
            value.clone_from(&other);
            assert_eq!(value, other);
            assert_eq!(value.precision(), metadata);
            assert_eq!(value.as_debug_verbose().0, &value);
            assert!(!format!("{:?}", value.as_debug_verbose()).is_empty());
            assert!(MpUint::with_capacity(8).capacity() >= 8);
        }
        1 => {
            assert_eq!(
                MpUint::with_precision_checked(a.clone(), precision).map(|value| value.to_string()),
                bounds
                    .fits(ra)
                    .then(|| ra.to_string())
                    .ok_or(MpError::PrecisionExceeded)
            );
            assert_integer(
                MpUint::with_precision_wrapping(a.clone(), precision),
                &bounds.wrap(ra),
            );
            assert_integer(
                MpUint::with_precision_saturating(a.clone(), precision),
                &bounds.saturate(ra),
            );
            assert!(MpUint::zero().is_zero());
            assert!(MpUint::one().is_one());
        }
        2 => {
            assert_eq!(precision.get(), bits);
            let p = Precision::new_bounded(bits).unwrap();
            assert_eq!(p.significant_bits(), Some(bits));
            assert!(!p.is_unlimited());
            assert!(Precision::Unlimited.is_unlimited());
            assert_eq!(Precision::Unlimited.significant_bits(), None);
            assert_eq!(
                AmbientPrecision::new_bounded(bits),
                Some(AmbientPrecision::Bounded(precision))
            );
            for invalid in [0, usize::MAX] {
                assert_eq!(BoundedPrecision::new(invalid), None);
                assert_eq!(Precision::new_bounded(invalid), None);
                assert_eq!(AmbientPrecision::new_bounded(invalid), None);
            }
            let before = PrecisionContext::active();
            #[cfg(feature = "std")]
            PrecisionContext::with_bounded(bits, || {
                assert_eq!(
                    PrecisionContext::active(),
                    AmbientPrecision::Bounded(precision)
                );
                assert_eq!(MpUint::with_capacity(8).precision(), p);
                assert_eq!(
                    MpUint::from_str_radix(&ra.to_string(), 10).is_ok(),
                    bounds.fits(ra)
                );
                PrecisionContext::with_unlimited(|| {
                    assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                    assert_eq!(
                        MpUint::from(input.parameter).precision(),
                        Precision::Unlimited
                    );
                });
                assert_eq!(
                    PrecisionContext::active(),
                    AmbientPrecision::Bounded(precision)
                );
            });
            assert_eq!(PrecisionContext::active(), before);
        }
        _ => {
            let b = MpUint::from_be_bytes(input.right);
            let rb = Integer::from_digits(input.right, Order::Msf);
            let mut destination = MpUint::zero_with_precision(precision);
            let sum = Integer::from(ra + &rb);
            if bounds.fits(&sum) {
                destination.assign_add(a, &b);
                assert_integer(&destination, &sum);
            }
            let difference = Integer::from(ra - &rb);
            if difference < 0 {
                let before = destination.clone();
                assert!(destination.assign_sub(a, &b));
                assert_eq!(destination, before);
            } else if bounds.fits(&difference) {
                assert!(!destination.assign_sub(a, &b));
                assert_integer(&destination, &difference);
            }
            let product = Integer::from(ra * &rb);
            if bounds.fits(&product) {
                destination.assign_mul(a, &b);
                assert_integer(&destination, &product);
            }
            let square = Integer::from(ra * ra);
            if bounds.fits(&square) {
                destination.assign_square(a);
                assert_integer(&destination, &square);
            }
            assert_eq!(destination.precision(), Precision::Bounded(precision));
        }
    }
}
