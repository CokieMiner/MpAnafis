//! Bounded and unlimited unsigned theory, powers, and roots against GMP.

use mp_anafis::{BoundedPrecision, MpError, MpUint, Precision};
use rug::{Integer, integer::IsPrime, ops::Pow};

use crate::{Bounds, TheoryReference, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, operation: u8, parameter: u16) {
    let width = usize::from(parameter % 512) + 1;
    let finite = parameter & 0x4000 == 0;
    let bounds = Bounds {
        bits: finite.then_some(width),
        signed: false,
    };
    let precision = BoundedPrecision::new(width).unwrap();
    let a = if finite {
        MpUint::with_precision_wrapping(a.clone(), precision)
    } else {
        a.clone()
    };
    let b = if finite {
        MpUint::with_precision_wrapping(b.clone(), precision)
    } else {
        b.clone()
    };
    let ra = bounds.wrap(ra);
    let rb = bounds.wrap(rb);
    match operation % 10 {
        0 => {
            let g = ra.clone().gcd(&rb);
            let l = ra.clone().lcm(&rb);
            assert_integer(a.gcd(&b), &g);
            assert_optional(a.lcm(&b), bounds.fits(&l).then(|| l.clone()));
            let pair = a.gcd_lcm(&b);
            assert_eq!(pair.is_some(), bounds.fits(&l));
            if let Some((actual_g, actual_l)) = pair {
                assert_integer(actual_g, &g);
                assert_integer(actual_l, &l);
            }
            assert_eq!(a.is_coprime(&b), g == 1);
        }
        1 => {
            if rb == 0 {
                assert_eq!(a.extended_gcd(&b), None);
                return;
            }
            let (g, x, y) = a.extended_gcd(&b).unwrap();
            let rg = ra.clone().gcd(&rb);
            assert_integer(g, &rg);
            let x = Integer::from_str_radix(&x.to_string(), 10).unwrap();
            let y = Integer::from_str_radix(&y.to_string(), 10).unwrap();
            assert_eq!((&ra * x) % &rb, rg.clone() % &rb);
            if ra != 0 {
                assert_eq!((&rb * y) % &ra, rg % &ra);
            }
        }
        2 => {
            assert_optional(a.isqrt(), Some(ra.clone().sqrt()));
            assert_eq!(a.is_perfect_square(), ra.is_perfect_square());
        }
        3 => {
            let degree = u32::from(parameter % 11);
            let expected = (degree != 0).then(|| ra.clone().root(degree));
            assert_optional(a.nth_root(degree), expected);
        }
        4 => {
            let (s, r) = a.sqrt_rem().unwrap();
            let (rs, rr) = ra.clone().sqrt_rem(Integer::new());
            assert_integer(s, &rs);
            assert_integer(r, &rr);
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
            // Large primality inputs are limited to 512 bits; larger magnitudes project to u64.
            let n = if ra.significant_bits() <= 512 {
                ra
            } else {
                Integer::from(ra.to_u64_wrapping())
            };
            let value = MpUint::from_str_radix(&n.to_string(), 10).unwrap();
            let classification = n.is_probably_prime(25);
            let prime = value.is_prime();
            if n.to_u64().is_some() {
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
            let expected = rb.is_odd().then(|| ra.jacobi(&rb));
            assert_eq!(a.jacobi_symbol(&b).map(i32::from), expected);
        }
        8 => {
            let n = Integer::from(ra.to_u16_wrapping());
            let value = if finite {
                MpUint::with_precision_wrapping(n.to_u16().unwrap(), precision)
            } else {
                MpUint::from(n.to_u16().unwrap())
            };
            let n = bounds.wrap(&n);
            let next = n.clone().next_prime();
            assert_optional(value.next_prime(), bounds.fits(&next).then_some(next));
            let mut previous = n - 1_u32;
            while previous >= 2 && previous.is_probably_prime(25) == IsPrime::No {
                previous -= 1;
            }
            assert_optional(value.prev_prime(), (previous >= 2).then_some(previous));
        }
        _ => {
            let n = ra.to_u16_wrapping() % 1024;
            let phi = MpUint::from(n).euler_phi();
            if n == 0 {
                assert_eq!(phi, None);
            } else if let Some(phi) = phi {
                assert_integer(phi, &TheoryReference::phi(n).unwrap());
            }
            let n = u32::from(parameter % 129);
            let expected = Integer::from(Integer::factorial(n));
            assert_integer(MpUint::factorial(n, Precision::Unlimited), &expected);
            if bounds.fits(&expected) {
                assert_integer(MpUint::factorial(n, a.precision()), &expected);
            }
        }
    }
}
