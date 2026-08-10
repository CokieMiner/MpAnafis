#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "int/support.rs"]
mod support;

#[path = "int/arithmetic.rs"]
mod arithmetic;
#[path = "int/bitwise.rs"]
mod bitwise;
#[path = "int/conversion.rs"]
mod conversion;
#[path = "int/division.rs"]
mod division;
#[path = "int/theory.rs"]
mod theory;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let part_size = data.len() / 3;
    let (left, rest) = data.split_at(part_size);
    let (right, selector) = rest.split_at(rest.len().min(part_size));

    let op = selector.first().copied().unwrap_or(0);
    let (left_hex, right_hex) = support::parse_signed_hex_pair(left, right, op);
    let (arbi_a, arbi_b, rug_a, rug_b) = support::int_operands(&left_hex, &right_hex);

    match op % 5 {
        0 => arithmetic::fuzz_all(&arbi_a, &arbi_b, &rug_a, &rug_b, op),
        1 => division::fuzz_all(&arbi_a, &arbi_b, &rug_a, &rug_b, op),
        2 => bitwise::fuzz_all(&arbi_a, &arbi_b, &rug_a, &rug_b, op),
        3 => conversion::fuzz_all(&arbi_a, op),
        4 => theory::fuzz_all(&arbi_a, &arbi_b, &rug_a, &rug_b, op),
        _ => unreachable!(),
    }
});
