//! Signed constructors, storage, absolute value, and fused assignment metadata.

use mp_anafis::{AmbientPrecision, BoundedPrecision, MpError, MpInt, Precision, PrecisionContext};
use rug::{Integer, integer::Order};

use crate::{Bounds, Input, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpInt, ra: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let precision = BoundedPrecision::new(bits).unwrap();
    let bounds = Bounds {
        bits: Some(bits),
        signed: true,
    };
    match input.operation % 4 {
        0 => {
            let mut value = MpInt::new(a.clone());
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
            let mut other = MpInt::zero_with_precision(precision);
            value.swap(&mut other);
            assert!(value.is_zero());
            assert_eq!(value.precision(), Precision::Bounded(precision));
            assert_integer(&other, ra);
            assert_eq!(other.precision(), metadata);
            value.clone_from(&other);
            assert_eq!(value, other);
            assert_eq!(value.precision(), metadata);
            assert_eq!(value.as_debug_verbose().0, &value);
            assert!(!format!("{:?}", value.as_debug_verbose()).is_empty());
            assert!(MpInt::with_capacity(8).capacity() >= 8);
        }
        1 => {
            assert_eq!(
                MpInt::with_precision_checked(a.clone(), precision).map(|value| value.to_string()),
                bounds
                    .fits(ra)
                    .then(|| ra.to_string())
                    .ok_or(MpError::PrecisionExceeded)
            );
            assert_integer(
                MpInt::with_precision_wrapping(a.clone(), precision),
                &bounds.wrap(ra),
            );
            assert_integer(
                MpInt::with_precision_saturating(a.clone(), precision),
                &bounds.saturate(ra),
            );
            let mut value = MpInt::with_precision_wrapping(a.clone(), precision);
            let absolute = bounds.wrap(ra).abs();
            assert_optional(
                value.checked_abs(),
                bounds.fits(&absolute).then(|| absolute.clone()),
            );
            assert_integer(value.unsigned_abs(), &absolute);
            if bounds.fits(&absolute) {
                value.abs_assign();
                assert_integer(value, &absolute);
            }
            assert!(MpInt::zero().is_zero());
            assert!(MpInt::one().is_one());
            assert!(MpInt::minus_one().is_minus_one());
        }
        2 => {
            assert_eq!(precision.get(), bits);
            let p = Precision::new_bounded(bits).unwrap();
            assert_eq!(p.significant_bits(), Some(bits));
            assert!(!p.is_unlimited());
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
                assert_eq!(MpInt::with_capacity(8).precision(), p);
                assert_eq!(
                    MpInt::from_str_radix(&ra.to_string(), 10).is_ok(),
                    bounds.fits(ra)
                );
                PrecisionContext::with_unlimited(|| {
                    assert_eq!(PrecisionContext::active(), AmbientPrecision::Unlimited);
                    assert_eq!(
                        MpInt::from(input.parameter).precision(),
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
            let mut rb = Integer::from_digits(input.right, Order::Msf);
            if input.flags & 0x40 != 0 {
                rb = -rb;
            }
            let b = MpInt::from_str_radix(&rb.to_string(), 10).unwrap();
            let mut destination = MpInt::zero_with_precision(precision);
            let sum = Integer::from(ra + &rb);
            if bounds.fits(&sum) {
                destination.assign_add(a, &b);
                assert_integer(&destination, &sum);
            }
            let difference = Integer::from(ra - &rb);
            if bounds.fits(&difference) {
                destination.assign_sub(a, &b);
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
