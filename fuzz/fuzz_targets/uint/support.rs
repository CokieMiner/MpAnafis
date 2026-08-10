//! Helper utilities for `ArbiUint` operand parsing.

use arbi_anafis::ArbiUint;
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

pub fn uint_operands(left_hex: &str, right_hex: &str) -> (ArbiUint, ArbiUint, Integer, Integer) {
    (
        ArbiUint::from_str_radix(left_hex, 16).unwrap(),
        ArbiUint::from_str_radix(right_hex, 16).unwrap(),
        Integer::from_str_radix(left_hex, 16).unwrap(),
        Integer::from_str_radix(right_hex, 16).unwrap(),
    )
}
