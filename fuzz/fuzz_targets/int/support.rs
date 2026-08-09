//! Helper utilities for `MpInt` operand parsing.

use mp_anafis::MpInt;
use rug::Integer;

pub fn parse_signed_hex_pair(
    left: &[u8],
    right: &[u8],
    selector_flags: u8,
) -> (String, String) {
    let mut left_hex = hex::encode(left);
    let mut right_hex = hex::encode(right);
    if left_hex.is_empty() {
        left_hex = "0".to_string();
    }
    if right_hex.is_empty() {
        right_hex = "0".to_string();
    }
    if selector_flags & 0x80 != 0 {
        left_hex = format!("-{left_hex}");
    }
    if selector_flags & 0x40 != 0 {
        right_hex = format!("-{right_hex}");
    }
    (left_hex, right_hex)
}

pub fn int_operands(left_hex: &str, right_hex: &str) -> (MpInt, MpInt, Integer, Integer) {
    (
        MpInt::from_str_radix(left_hex, 16).unwrap(),
        MpInt::from_str_radix(right_hex, 16).unwrap(),
        Integer::from_str_radix(left_hex, 16).unwrap(),
        Integer::from_str_radix(right_hex, 16).unwrap(),
    )
}
