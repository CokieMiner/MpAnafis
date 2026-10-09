//! Signed fused arithmetic and two's-complement word decomposition.

use mp_anafis::{BoundedPrecision, MpError, MpInt, MpUint};
use rug::{Integer, integer::Order};

use crate::{Bounds, Input, assert_integer};

pub fn fuzz_all(a: &MpInt, b: &MpInt, ra: &Integer, rb: &Integer, input: &Input<'_>) {
    let bits = usize::from(input.parameter % 512) + 1;
    let left = Bounds {
        bits: (input.flags & 1 == 0).then_some(bits),
        signed: true,
    };
    let right = Bounds {
        bits: (input.flags & 2 == 0).then_some(usize::from(input.parameter >> 8) + 1),
        signed: true,
    };
    let carrying = Bounds {
        bits: (input.flags & 4 == 0).then_some(usize::from(input.parameter % 256) + 1),
        signed: true,
    };
    let pair_bits = left
        .bits
        .zip(right.bits)
        .map(|(left, right)| left.max(right));
    let combined = if input.operation.is_multiple_of(3) {
        pair_bits
    } else {
        pair_bits
            .zip(carrying.bits)
            .map(|(pair, carry)| pair.max(carry))
    };
    let bounds = Bounds {
        bits: combined,
        signed: true,
    };
    let mut raw_carry = Integer::from_digits(input.modulus, Order::Msf);
    let mut carry = MpInt::from(MpUint::from_be_bytes(input.modulus));
    if input.flags & 0x20 != 0 {
        raw_carry = -raw_carry;
        carry = -carry;
    }
    let construct = |value: &MpInt, bits: Option<usize>| {
        bits.map_or_else(
            || value.clone(),
            |bits| {
                MpInt::with_precision_wrapping(value.clone(), BoundedPrecision::new(bits).unwrap())
            },
        )
    };
    let (a, b, carry) = (
        construct(a, left.bits),
        construct(b, right.bits),
        construct(&carry, carrying.bits),
    );
    let (ra, rb, rc) = (left.wrap(ra), right.wrap(rb), carrying.wrap(&raw_carry));
    let check_metadata = |pair: &(MpInt, MpInt)| {
        for value in [&pair.0, &pair.1] {
            assert_eq!(value.precision().significant_bits(), combined);
        }
    };
    let product = Integer::from(&ra * &rb);
    let (actual, exact) = match input.operation % 3 {
        0 => {
            let actual = a.widening_mul(&b);
            if bounds.bits.is_some() {
                let tried = a.try_widening_mul(&b).unwrap();
                check_metadata(&tried);
                assert_eq!(tried, actual);
            } else {
                assert_eq!(a.try_widening_mul(&b), Err(MpError::WidthRequired));
            }
            (actual, product)
        }
        1 => {
            let actual = a.carrying_mul(&b, &carry);
            if bounds.bits.is_some() {
                let tried = a.try_carrying_mul(&b, &carry).unwrap();
                check_metadata(&tried);
                assert_eq!(tried, actual);
            } else {
                assert_eq!(a.try_carrying_mul(&b, &carry), Err(MpError::WidthRequired));
            }
            (actual, product + &rc)
        }
        _ => {
            let fused = Integer::from(&product + &rc);
            if bounds.fits(&fused) {
                assert_integer(a.mul_add(&b, &carry), &fused);
            }
            assert_integer(a.midpoint(&b), &(Integer::from(&ra + &rb) / 2_u8));
            (a.carrying_mul_add(&b, &carry, &a), fused + &ra)
        }
    };
    check_metadata(&actual);
    assert_integer(actual.0, &bounds.wrap(&exact));
    let high = bounds.bits.map_or_else(Integer::new, |bits| {
        bounds.wrap(&Integer::from(&exact >> bits))
    });
    assert_integer(actual.1, &high);
}
