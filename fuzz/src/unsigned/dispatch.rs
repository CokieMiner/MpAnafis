//! Unsigned category dispatch with independently decoded operation controls.

use crate::Input;

use super::{
    arithmetic, bitwise, bounded, combined, conversion, division, metadata, modular,
    parse_hex_pair, properties, theory, traits, uint_operands,
};

/// Reachable operation counts in category order.
pub const OPERATIONS: [u8; 11] = [6, 3, 5, 5, 10, 7, 6, 3, 4, 4, 5];

pub fn run(data: &[u8]) {
    let Some(input) = Input::parse(data) else {
        return;
    };
    let (left, right) = parse_hex_pair(input.left, input.right);
    let (a, b, ra, rb) = uint_operands(&left, &right);
    match input.category {
        0 => arithmetic(&a, &b, &ra, &rb, input.operation),
        1 => division(&a, &b, &ra, &rb, &input),
        2 => bitwise(&a, &b, &ra, &rb, input.operation, input.parameter),
        3 => conversion(&a, &ra, input.operation, input.parameter, input.left),
        4 => theory(&a, &b, &ra, &rb, input.operation, input.parameter),
        5 => modular(&a, &b, &ra, &rb, &input),
        6 => bounded(&a, &b, &ra, &rb, &input),
        7 => combined(&a, &b, &ra, &rb, &input),
        8 => properties(&a, &b, &ra, &rb, &input),
        9 => metadata(&a, &ra, &input),
        _ => traits(&a, &b, &ra, &rb, &input),
    }
}
