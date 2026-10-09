//! Bounded division modes and zero-divisor rejection against GMP.

use mp_anafis::{BoundedPrecision, MpUint};
use rug::{Integer, ops::DivRounding};

use crate::{Bounds, Input, assert_integer, assert_optional};

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let bounds = Bounds {
        bits: (input.flags & 1 == 0).then_some(bits),
        signed: false,
    };
    let precision = BoundedPrecision::new(bits).unwrap();
    let a = if bounds.bits.is_some() {
        MpUint::with_precision_wrapping(a.clone(), precision)
    } else {
        a.clone()
    };
    let b = if bounds.bits.is_some() {
        MpUint::with_precision_wrapping(b.clone(), precision)
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
    let pair = (rb != 0).then(|| ra.clone().div_rem(rb.clone()));
    let quotient = pair.as_ref().map(|(q, _)| q.clone());
    let remainder = pair.as_ref().map(|(_, r)| r.clone());
    macro_rules! scalar {
        ($checked:ident, $strict:ident, $expected:expr) => {{
            let expected = $expected;
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
            scalar!(checked_div_trunc, div_trunc, quotient.clone());
            scalar!(checked_rem_trunc, rem_trunc, remainder.clone());
            scalar!(checked_div_euclid, div_euclid, quotient);
            scalar!(checked_rem_euclid, rem_euclid, remainder);
            let actual = a.div_rem_euclid(&b);
            assert_eq!(actual.is_some(), pair.is_some());
            if let Some((q, r)) = actual {
                let (rq, rr) = pair.unwrap();
                assert_integer(q, &rq);
                assert_integer(r, &rr);
            }
        }
        _ => {
            scalar!(checked_div_floor, div_floor, quotient);
            scalar!(checked_mod_floor, mod_floor, remainder);
            scalar!(
                checked_div_ceil,
                div_ceil,
                (rb != 0).then(|| ra.clone().div_ceil(&rb))
            );
            let actual = a.div_rem_floor(&b);
            assert_eq!(actual.is_some(), pair.is_some());
            if let Some((q, r)) = actual {
                let (rq, rr) = pair.unwrap();
                assert_integer(q, &rq);
                assert_integer(r, &rr);
            }
        }
    }
}
