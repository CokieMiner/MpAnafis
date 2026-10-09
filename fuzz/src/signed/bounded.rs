//! Signed arithmetic policies with independent widths and asymmetric endpoints.

use mp_anafis::{BoundedPrecision, MpError, MpInt, Precision};
use rug::Integer;

use crate::{Bounds, Input, PolicyResults, assert_integer};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let left_bits = usize::from(input.parameter % 512) + 1;
    let right_bits = usize::from(input.parameter >> 8) + 1;
    let left = Bounds {
        bits: (input.flags & 1 == 0).then_some(left_bits),
        signed: true,
    };
    let right = Bounds {
        bits: (input.flags & 2 == 0).then_some(right_bits),
        signed: true,
    };
    let a = left.bits.map_or_else(
        || a.clone(),
        |bits| MpInt::with_precision_wrapping(a.clone(), BoundedPrecision::new(bits).unwrap()),
    );
    let b = right.bits.map_or_else(
        || b.clone(),
        |bits| MpInt::with_precision_wrapping(b.clone(), BoundedPrecision::new(bits).unwrap()),
    );
    let ra = left.wrap(ra);
    let rb = right.wrap(rb);
    let bounds = Bounds {
        bits: left.bits.zip(right.bits).map(|(a, b)| a.max(b)),
        signed: true,
    };
    let rejected = input.operation % 6 == 4
        && rb == -1
        && bounds
            .bits
            .is_some_and(|bits| ra == -(Integer::from(1) << (bits - 1)));
    macro_rules! check {
        ($exact:expr, $checked:ident, $tried:ident, $wrapping:ident, $saturating:ident, $overflowing:ident, $strict:ident) => {{
            let exact: Option<Integer> = $exact;
            if !rejected && exact.as_ref().is_some_and(|value| bounds.fits(value)) {
                assert_integer(a.$strict(&b), exact.as_ref().unwrap());
            }
            let results = PolicyResults {
                checked: a.$checked(&b),
                tried: a.$tried(&b),
                wrapping: Some(a.$wrapping(&b)),
                saturating: a.$saturating(&b),
                overflowing: a.$overflowing(&b),
            };
            let precision = bounds
                .bits
                .and_then(Precision::new_bounded)
                .unwrap_or(Precision::Unlimited);
            for value in [
                results.checked.as_ref(),
                results.tried.as_ref().ok(),
                results.wrapping.as_ref(),
                Some(&results.saturating),
                Some(&results.overflowing.0),
            ]
            .into_iter()
            .flatten()
            {
                assert_eq!(value.precision(), precision);
            }
            bounds.check(results, exact, rejected);
        }};
    }
    match input.operation % 6 {
        0 => check!(
            Some(Integer::from(&ra + &rb)),
            checked_add,
            try_add,
            wrapping_add,
            saturating_add,
            overflowing_add,
            strict_add
        ),
        1 => check!(
            Some(Integer::from(&ra - &rb)),
            checked_sub,
            try_sub,
            wrapping_sub,
            saturating_sub,
            overflowing_sub,
            strict_sub
        ),
        2 => check!(
            Some(Integer::from(&ra * &rb)),
            checked_mul,
            try_mul,
            wrapping_mul,
            saturating_mul,
            overflowing_mul,
            strict_mul
        ),
        3 => check!(
            (rb != 0).then(|| Integer::from(&ra / &rb)),
            checked_div,
            try_div,
            wrapping_div,
            saturating_div,
            overflowing_div,
            strict_div
        ),
        4 => check!(
            (rb != 0).then(|| Integer::from(&ra % &rb)),
            checked_rem,
            try_rem,
            wrapping_rem,
            saturating_rem,
            overflowing_rem,
            strict_rem
        ),
        _ => {
            let shift = usize::from(input.parameter % 1024);
            let exact = Integer::from(&ra << shift);
            let checked = left.fits(&exact).then(|| exact.to_string());
            assert_eq!(a.checked_shl(shift).map(|value| value.to_string()), checked);
            assert_eq!(
                a.try_shl(shift).map(|value| value.to_string()),
                checked.ok_or(MpError::Overflow)
            );
            assert_integer(a.wrapping_shl(shift), &left.wrap(&exact));
            assert_integer(a.saturating_shl(shift), &left.saturate(&exact));
            let (value, overflow) = a.overflowing_shl(shift);
            assert_integer(value, &left.wrap(&exact));
            assert_eq!(overflow, !left.fits(&exact));
        }
    }
}
