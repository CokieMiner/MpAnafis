//! Signed category dispatch with independently decoded operation controls.

use crate::Input;

use super::{
    arithmetic, bitwise, bounded, combined, conversion, division, int_operands, metadata, modular,
    parse_signed_hex_pair, properties, theory, traits,
};

/// Reachable operation counts in category order.
pub const OPERATIONS: [u8; 11] = [6, 3, 5, 5, 10, 7, 6, 3, 4, 4, 5];

pub fn run(data: &[u8]) {
    let Some(input) = Input::parse(data) else {
        return;
    };
    let (left, right) = parse_signed_hex_pair(input.left, input.right, input.flags);
    let (a, b, ra, rb) = int_operands(&left, &right);
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
