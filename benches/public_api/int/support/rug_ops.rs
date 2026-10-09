//! Composed Rug equivalents for API operations without single-call counterparts.

use rug::{Integer, integer::Order};

/// Splits an exact product into unsigned low and high words.
pub fn unsigned_words(value: Integer, bits: usize) -> (Integer, Integer) {
    let width = u32::try_from(bits).expect("benchmark width fits Rug");
    let upper = Integer::from(&value >> width);
    (value.keep_bits(width), upper)
}

/// Splits an exact signed product into signed two's-complement half-words.
pub fn signed_words(value: Integer, bits: usize) -> (Integer, Integer) {
    let width = u32::try_from(bits).expect("benchmark width fits Rug");
    let upper = Integer::from(&value >> width).keep_signed_bits(width);
    (value.keep_signed_bits(width), upper)
}

/// Produces the minimal signed big-endian two's-complement encoding.
pub fn signed_be_bytes(value: &Integer) -> Vec<u8> {
    let signed_bits = if value < &0 {
        Integer::from(!value).significant_bits()
    } else {
        value.significant_bits()
    }
    .checked_add(1)
    .expect("signed width fits u32");
    let bytes = usize::try_from(signed_bits.div_ceil(8)).expect("byte count fits usize");
    let width = u32::try_from(bytes.checked_mul(8).expect("byte width fits usize"))
        .expect("byte width fits Rug");
    let residue = value.clone().keep_bits(width);
    let mut output = vec![0; bytes];
    residue.write_digits(&mut output, Order::MsfBe);
    output
}

/// Imports the same signed two's-complement encoding consumed by `MpInt`.
pub fn from_signed_be_bytes(bytes: &[u8]) -> Integer {
    let value = Integer::from_digits(bytes, Order::MsfBe);
    let width = u32::try_from(bytes.len().checked_mul(8).expect("byte width fits usize"))
        .expect("byte width fits Rug");
    value.keep_signed_bits(width)
}

/// Rounds to nearest, ties to even, before GMP's truncating float export.
/// The explicit integer rounding is part of the timed composed reference.
pub fn nearest_f64(value: &Integer) -> Option<f64> {
    if value.significant_bits() > 1024 {
        return None;
    }
    let result = rounded_integer(value, 53).to_f64();
    result.is_finite().then_some(result)
}

/// Rounds once to the 24-bit significand before GMP's float export.
pub fn nearest_f32(value: &Integer) -> Option<f32> {
    if value.significant_bits() > 128 {
        return None;
    }
    let result = rounded_integer(value, 24).to_f32();
    result.is_finite().then_some(result)
}

fn rounded_integer(value: &Integer, precision: u32) -> Integer {
    let magnitude = value.clone().abs();
    let shift = magnitude.significant_bits().saturating_sub(precision);
    if shift == 0 {
        return value.clone();
    }
    let mut high = Integer::from(&magnitude >> shift);
    let remainder = magnitude.keep_bits(shift);
    let halfway = Integer::from(1) << shift.checked_sub(1).expect("positive shift");
    if remainder > halfway || (remainder == halfway && high.is_odd()) {
        high += 1;
    }
    high <<= shift;
    if value < &0 { -high } else { high }
}
