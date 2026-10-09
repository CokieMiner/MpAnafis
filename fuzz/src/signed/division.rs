//! Signed bounded division modes, quotient overflow, and divisibility against GMP.

use mp_anafis::{BoundedPrecision, MpInt};
use rug::{Integer, ops::DivRounding};

use crate::{Bounds, Input, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let bounds = Bounds {
        bits: (input.flags & 1 == 0).then_some(bits),
        signed: true,
    };
    let precision = BoundedPrecision::new(bits).unwrap();
    let a = if bounds.bits.is_some() {
        MpInt::with_precision_wrapping(a.clone(), precision)
    } else {
        a.clone()
    };
    let b = if bounds.bits.is_some() {
        MpInt::with_precision_wrapping(b.clone(), precision)
    } else {
        b.clone()
    };
    let ra = bounds.wrap(ra);
    let rb = bounds.wrap(rb);
    if input.operation % 3 == 2 {
        let divides = if rb == 0 {
            ra == 0
        } else {
            Integer::from(&ra % &rb) == 0
        };
        assert_eq!(a.is_divisible_by(&b), divides);
        assert_eq!(b.is_divisor_of(&a), divides);
        return;
    }
    // MIN / -1 rejects all bounded quotient and remainder variants.
    let valid = rb != 0 && bounds.fits(&Integer::from(&ra / &rb));
    macro_rules! scalar {
        ($checked:ident, $strict:ident, $expected:expr) => {{
            let expected: Option<Integer> = valid.then(|| $expected);
            assert_optional(a.$checked(&b), expected.clone());
            if let Some(expected) = expected {
                let result = a.$strict(&b);
                assert_integer(&result, &expected);
                assert_eq!(result.precision(), a.precision());
            }
        }};
    }
    match input.operation % 3 {
        0 => {
            scalar!(checked_div_trunc, div_trunc, Integer::from(&ra / &rb));
            scalar!(checked_rem_trunc, rem_trunc, Integer::from(&ra % &rb));
            let expected = valid.then(|| ra.clone().div_rem_euc(rb.clone()));
            scalar!(
                checked_div_euclid,
                div_euclid,
                expected.as_ref().unwrap().0.clone()
            );
            scalar!(
                checked_rem_euclid,
                rem_euclid,
                expected.as_ref().unwrap().1.clone()
            );
            let actual = a.div_rem_euclid(&b);
            assert_eq!(actual.is_some(), valid);
            if let Some((q, r)) = actual {
                let (rq, rr) = expected.unwrap();
                assert_integer(q, &rq);
                assert_integer(r, &rr);
            }
        }
        _ => {
            let expected = valid.then(|| ra.clone().div_rem_floor(rb.clone()));
            scalar!(
                checked_div_floor,
                div_floor,
                expected.as_ref().unwrap().0.clone()
            );
            scalar!(
                checked_mod_floor,
                mod_floor,
                expected.as_ref().unwrap().1.clone()
            );
            scalar!(checked_div_ceil, div_ceil, ra.clone().div_ceil(&rb));
            let actual = a.div_rem_floor(&b);
            assert_eq!(actual.is_some(), valid);
            if let Some((q, r)) = actual {
                let (rq, rr) = expected.unwrap();
                assert_integer(q, &rq);
                assert_integer(r, &rr);
            }
        }
    }
}
