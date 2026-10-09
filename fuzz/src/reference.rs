//! Shared GMP-based result and byte-encoding checks.

use core::fmt::Display;

use rug::{Integer, integer::Order};

pub fn assert_integer(actual: impl Display, expected: &Integer) {
    assert_eq!(actual.to_string(), expected.to_string());
}

pub fn assert_optional(actual: Option<impl Display>, expected: Option<Integer>) {
    assert_eq!(
        actual.map(|value| value.to_string()),
        expected.map(|value| value.to_string())
    );
}

/// Computes the public modular contracts over nonnegative magnitudes using GMP.
pub fn modular(
    a: &Integer,
    b: &Integer,
    modulus: &Integer,
    operation: u8,
    exponent: i32,
) -> Option<Integer> {
    if modulus == &0 {
        return None;
    }
    match operation % 7 {
        0 => Some((Integer::from(a + b)) % modulus),
        1 => Some((Integer::from(a - b)).modulo(modulus)),
        2 => Some((Integer::from(a * b)) % modulus),
        3 => a.clone().pow_mod(&Integer::from(exponent), modulus).ok(),
        4 => a.clone().invert(modulus).ok(),
        5 => Some(a.clone() % modulus),
        _ => {
            if modulus.is_even() {
                return None;
            }
            let radix_bits = modulus.significant_bits().div_ceil(usize::BITS) * usize::BITS;
            let inverse = (Integer::from(1) << radix_bits)
                .invert(modulus)
                .expect("odd modulus has a radix inverse");
            Some((Integer::from(a * b) * inverse) % modulus)
        }
    }
}

/// Encodes a minimal signed two's-complement value using GMP arithmetic.
pub fn signed_bytes(value: &Integer) -> Vec<u8> {
    if value == &0 {
        return Vec::new();
    }
    let bits = if value < &0 {
        (!value.clone()).significant_bits() + 1
    } else {
        value.significant_bits() + 1
    };
    let bytes = bits.div_ceil(8);
    let residue = if value < &0 {
        (Integer::from(1) << (bytes * 8)) + value
    } else {
        value.clone()
    };
    let mut result = residue.to_digits::<u8>(Order::Lsf);
    result.resize(usize::try_from(bytes).expect("byte count fits usize"), 0);
    result
}

pub fn float64(value: &Integer) -> Option<f64> {
    let rounded = rounded_float_integer(value, f64::MANTISSA_DIGITS).to_f64();
    rounded.is_finite().then_some(rounded)
}

pub fn float32(value: &Integer) -> Option<f32> {
    let rounded = rounded_float_integer(value, f32::MANTISSA_DIGITS).to_f32();
    rounded.is_finite().then_some(rounded)
}

/// Rounds the magnitude to an integer multiple of the target float's spacing.
/// GMP supplies the quotient and remainder; halfway cases select an even
/// significand. The rounded integer then converts exactly, or overflows.
fn rounded_float_integer(value: &Integer, precision: u32) -> Integer {
    let magnitude = value.clone().abs();
    let discarded = magnitude.significant_bits().saturating_sub(precision);
    if discarded == 0 {
        return value.clone();
    }
    let spacing = Integer::from(1) << discarded;
    let (mut significand, remainder) = magnitude.div_rem(spacing.clone());
    let twice_remainder = remainder * 2_u32;
    if twice_remainder > spacing || (twice_remainder == spacing && significand.is_odd()) {
        significand += 1;
    }
    let rounded = significand << discarded;
    if value < &0 { -rounded } else { rounded }
}
