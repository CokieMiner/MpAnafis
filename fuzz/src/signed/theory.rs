//! Signed magnitude roots, Bezout identities, and theory against GMP.

use mp_anafis::{BoundedPrecision, MpError, MpInt, Precision};
use rug::{Integer, integer::IsPrime, ops::Pow};

use crate::{Bounds, TheoryReference, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, operation: u8, parameter: u16) {
    if operation % 10 == 1 {
        if rb == &0 {
            assert_eq!(a.extended_gcd(b), None);
            return;
        }
        let (g, x, y) = a.extended_gcd(b).unwrap();
        let gcd = ra.clone().gcd(rb);
        assert_integer(g, &gcd);
        let x = Integer::from_str_radix(&x.to_string(), 10).unwrap();
        let y = Integer::from_str_radix(&y.to_string(), 10).unwrap();
        assert_eq!(ra * x + rb * y, gcd);
        return;
    }
    let width = usize::from(parameter % 512) + 1;
    let finite = parameter & 0x4000 == 0;
    let bounds = Bounds {
        bits: finite.then_some(width),
        signed: true,
    };
    let precision = BoundedPrecision::new(width).unwrap();
    let a = if finite {
        MpInt::with_precision_wrapping(a.clone(), precision)
    } else {
        a.clone()
    };
    let b = if finite {
        MpInt::with_precision_wrapping(b.clone(), precision)
    } else {
        b.clone()
    };
    let ra = bounds.wrap(ra);
    let rb = bounds.wrap(rb);
    match operation % 10 {
        0 => {
            let g = ra.clone().gcd(&rb);
            let l = ra.clone().lcm(&rb);
            if bounds.fits(&g) {
                assert_integer(a.gcd(&b), &g);
            }
            assert_optional(a.lcm(&b), bounds.fits(&l).then(|| l.clone()));
            let pair = a.gcd_lcm(&b);
            assert_eq!(pair.is_some(), bounds.fits(&g) && bounds.fits(&l));
            if let Some((actual_g, actual_l)) = pair {
                assert_integer(actual_g, &g);
                assert_integer(actual_l, &l);
            }
            assert_eq!(a.is_coprime(&b), g == 1);
        }
        2 => {
            assert_optional(a.checked_isqrt(), (ra >= 0).then(|| ra.clone().sqrt()));
            assert_eq!(a.is_perfect_square(), ra.clone().abs().is_perfect_square());
        }
        3 => {
            let degree = u32::from(parameter % 11);
            let expected = if degree == 0 {
                None
            } else {
                Some(ra.clone().abs().root(degree))
            }
            .filter(|root| bounds.fits(root));
            assert_optional(a.nth_root(degree), expected);
        }
        4 => {
            let (s, r) = ra.clone().abs().sqrt_rem(Integer::new());
            let pair = a.sqrt_rem();
            assert_eq!(pair.is_some(), bounds.fits(&s) && bounds.fits(&r));
            if let Some((actual_s, actual_r)) = pair {
                assert_integer(actual_s, &s);
                assert_integer(actual_r, &r);
            }
        }
        5 => {
            let exponent = u32::from(parameter % 16);
            let expected = Integer::from((&ra).pow(exponent));
            assert_optional(
                a.checked_pow(exponent),
                bounds.fits(&expected).then(|| expected.clone()),
            );
            assert_eq!(
                a.try_pow(exponent).map(|value| value.to_string()),
                bounds
                    .fits(&expected)
                    .then(|| expected.to_string())
                    .ok_or(MpError::Overflow)
            );
            if bounds.fits(&expected) {
                assert_integer(a.pow(exponent), &expected);
            }
            let square = Integer::from(&ra * &ra);
            if bounds.fits(&square) {
                assert_integer(a.square(), &square);
            }
        }
        6 => {
            let n = if ra.significant_bits() <= 512 {
                ra
            } else {
                Integer::from(ra.to_i64_wrapping())
            };
            let value = MpInt::from_str_radix(&n.to_string(), 10).unwrap();
            let classification = if n < 2 {
                IsPrime::No
            } else {
                n.is_probably_prime(25)
            };
            let prime = value.is_prime();
            if n.clone().abs().to_u64().is_some() {
                assert_eq!(prime, classification != IsPrime::No);
            } else {
                if classification == IsPrime::Yes {
                    assert!(prime);
                }
                if prime {
                    assert!(TheoryReference::probably_prime(&n, 1));
                }
            }
            let rounds = u32::from(parameter % 66);
            assert_eq!(
                value.is_probably_prime(rounds),
                TheoryReference::probably_prime(&n, rounds)
            );
        }
        7 => {
            let expected = (rb > 0 && rb.is_odd()).then(|| ra.jacobi(&rb));
            assert_eq!(a.jacobi_symbol(&b).map(i32::from), expected);
        }
        8 => {
            let n = Integer::from(ra.to_i16_wrapping());
            let value = if finite {
                MpInt::with_precision_wrapping(n.to_i16().unwrap(), precision)
            } else {
                MpInt::from(n.to_i16().unwrap())
            };
            let n = bounds.wrap(&n);
            let next = n.clone().max(Integer::from(1)).next_prime();
            assert_optional(value.next_prime(), bounds.fits(&next).then_some(next));
            let mut previous = n - 1_u32;
            while previous >= 2 && previous.is_probably_prime(25) == IsPrime::No {
                previous -= 1;
            }
            assert_optional(value.prev_prime(), (previous >= 2).then_some(previous));
        }
        _ => {
            let n = ra.to_i16_wrapping() % 1024;
            let phi = MpInt::from(n).euler_phi();
            if n <= 0 {
                assert_eq!(phi, None);
            } else if let Some(phi) = phi {
                assert_integer(
                    phi,
                    &TheoryReference::phi(u16::try_from(n).unwrap()).unwrap(),
                );
            }
            let n = u32::from(parameter % 129);
            let expected = Integer::from(Integer::factorial(n));
            assert_integer(MpInt::factorial(n, Precision::Unlimited), &expected);
            if bounds.fits(&expected) {
                assert_integer(MpInt::factorial(n, a.precision()), &expected);
            }
        }
    }
}
