//! Signed magnitude modular arithmetic with signed exponents and moduli against GMP.

use mp_anafis::{BoundedPrecision, MpInt};
use rug::{Integer, integer::Order};

use crate::{Bounds, Input, assert_optional, modular};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let bounds = Bounds {
        bits: (input.flags & 1 == 0).then_some(bits),
        signed: true,
    };
    let precision = BoundedPrecision::new(bits).unwrap();
    let mut rm = Integer::from_digits(input.modulus, Order::Msf);
    if input.flags & 0x20 != 0 {
        rm = -rm;
    }
    let rm = bounds.wrap(&rm);
    let modulus = MpInt::from_str_radix(&rm.to_string(), 10).unwrap();
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
    let modulus = if bounds.bits.is_some() {
        MpInt::with_precision_wrapping(modulus, precision)
    } else {
        modulus
    };
    let exponent =
        i32::from(input.parameter % 256) * if input.parameter & 0x8000 == 0 { 1 } else { -1 };
    let exponent = if bounds.bits.is_some() {
        MpInt::with_precision_wrapping(exponent, precision)
    } else {
        MpInt::from(exponent)
    };
    let result = match input.operation % 7 {
        0 => a.add_mod(&b, &modulus),
        1 => a.sub_mod(&b, &modulus),
        2 => a.mul_mod(&b, &modulus),
        3 => a.pow_mod(&exponent, &modulus),
        4 => a.invert(&modulus),
        5 => a.barrett_reduce(&modulus),
        _ => a.montgomery_mul(&b, &modulus),
    };
    if let Some(result) = &result {
        assert_eq!(result.precision(), a.precision());
    }
    let exponent = bounds
        .wrap(&Integer::from(
            i32::from(input.parameter % 256) * if input.parameter & 0x8000 == 0 { 1 } else { -1 },
        ))
        .to_i32()
        .unwrap();
    assert_optional(
        result,
        modular(
            &bounds.wrap(ra).abs(),
            &bounds.wrap(rb).abs(),
            &rm.abs(),
            input.operation,
            exponent,
        ),
    );
}
