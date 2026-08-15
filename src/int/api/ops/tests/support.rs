//! Shared constructors and bounds for operator properties.

use super::{BoundedPrecision, MpInt, MpUint};

pub fn bounded_uint(value: u128, bits: usize) -> MpUint {
    let width = BoundedPrecision::new(bits).expect("property widths are valid");
    MpUint::with_precision_checked(value, width).expect("generated unsigned value fits")
}

pub fn bounded_int(value: i128, bits: usize) -> MpInt {
    let width = BoundedPrecision::new(bits).expect("property widths are valid");
    MpInt::with_precision_checked(value, width).expect("generated signed value fits")
}

pub fn unsigned_max(bits: usize) -> u128 {
    let shift = u32::try_from(bits).expect("property width fits u32");
    1_u128
        .checked_shl(shift)
        .expect("property width is below 128")
        .wrapping_sub(1)
}

pub fn signed_max(bits: usize) -> i128 {
    let magnitude_bits = bits.wrapping_sub(1);
    let shift = u32::try_from(magnitude_bits).expect("property width fits u32");
    1_i128
        .checked_shl(shift)
        .expect("property width is below 128")
        .wrapping_sub(1)
}

pub fn signed_min(bits: usize) -> i128 {
    signed_max(bits).wrapping_add(1).wrapping_neg()
}
