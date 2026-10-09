//! Unsigned modular arithmetic with bounded and unlimited precision against GMP.

use mp_anafis::{BoundedPrecision, MpUint};
use rug::{Integer, integer::Order};

use crate::{Bounds, Input, assert_optional, modular};

pub fn fuzz_all(a: &MpUint, b: &MpUint, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let bounds = Bounds {
        bits: (input.flags & 1 == 0).then_some(bits),
        signed: false,
    };
    let precision = BoundedPrecision::new(bits).unwrap();
    let rm = bounds.wrap(&Integer::from_digits(input.modulus, Order::Msf));
    let modulus = MpUint::from_str_radix(&rm.to_string(), 10).unwrap();
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
    let modulus = if bounds.bits.is_some() {
        MpUint::with_precision_wrapping(modulus, precision)
    } else {
        modulus
    };
    let exponent = input.parameter % 256;
    let exponent = if bounds.bits.is_some() {
        MpUint::with_precision_wrapping(exponent, precision)
    } else {
        MpUint::from(exponent)
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
        .wrap(&Integer::from(input.parameter % 256))
        .to_i32()
        .unwrap();
    assert_optional(
        result,
        modular(
            &bounds.wrap(ra),
            &bounds.wrap(rb),
            &rm,
            input.operation,
            exponent,
        ),
    );
}
