#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "uint/support.rs"]
mod support;

#[path = "uint/arithmetic.rs"]
mod arithmetic;
#[path = "uint/bitwise.rs"]
mod bitwise;
#[path = "uint/conversion.rs"]
mod conversion;
#[path = "uint/division.rs"]
mod division;
#[path = "uint/theory.rs"]
mod theory;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let part_size = data.len() / 3;
    let (left, rest) = data.split_at(part_size);
    let (right, selector) = rest.split_at(rest.len().min(part_size));

    let (left_hex, right_hex) = support::parse_hex_pair(left, right);
    let (mp_a, mp_b, rug_a, rug_b) = support::uint_operands(&left_hex, &right_hex);

    let op = selector.first().copied().unwrap_or(0);

    match op % 5 {
        0 => arithmetic::fuzz_all(&mp_a, &mp_b, &rug_a, &rug_b, op),
        1 => division::fuzz_all(&mp_a, &mp_b, &rug_b, op),
        2 => bitwise::fuzz_all(&mp_a, &mp_b, &rug_a, &rug_b, op),
        3 => conversion::fuzz_all(&mp_a, op),
        4 => theory::fuzz_all(&mp_a, &mp_b, &rug_a, &rug_b, op),
        _ => unreachable!(),
    }
});
