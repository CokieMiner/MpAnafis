//! Helper utilities for `MpUint` operand parsing.

use mp_anafis::MpUint;
use rug::Integer;

pub fn parse_hex_pair(left: &[u8], right: &[u8]) -> (String, String) {
    let mut left_hex = hex::encode(left);
    let mut right_hex = hex::encode(right);
    if left_hex.is_empty() {
        left_hex = "0".to_string();
    }
    if right_hex.is_empty() {
        right_hex = "0".to_string();
    }
    (left_hex, right_hex)
}

pub fn uint_operands(left_hex: &str, right_hex: &str) -> (MpUint, MpUint, Integer, Integer) {
    (
        MpUint::from_str_radix(left_hex, 16).unwrap(),
        MpUint::from_str_radix(right_hex, 16).unwrap(),
        Integer::from_str_radix(left_hex, 16).unwrap(),
        Integer::from_str_radix(right_hex, 16).unwrap(),
    )
}
