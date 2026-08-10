//! Deterministic bounded operands shared by the policy benchmarks.

use arbi_anafis::{ArbiUint, BoundedPrecision};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::support::{SAMPLES, arbi_uint};

/// One representative width for exceptional policy branches.
pub const EDGE_WIDTH: [usize; 1] = [1_024];

/// Arithmetic operation whose operand shape is requested.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Remainder.
    Rem,
}

/// Whether operands exercise the successful or exceptional policy path.
#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    /// The mathematical result fits and the divisor is non-zero.
    Success,
    /// Overflow, underflow, or a zero divisor, according to the operation.
    Edge,
}

/// Generates bounded Arbi operands for one policy operation and scenario.
#[must_use]
pub fn arbi_pairs(
    bits: usize,
    operation: Operation,
    scenario: Scenario,
) -> Vec<(ArbiUint, ArbiUint)> {
    let (left_bits, right_bits) = operand_widths(bits, operation, scenario);
    let width = BoundedPrecision::new(bits).expect("policy benchmark widths are valid");

    (0..SAMPLES)
        .map(|index| {
            let left = ArbiUint::with_precision_checked(
                arbi_uint(left_bits, 42_u32.wrapping_add(index)),
                width,
            )
            .expect("the generated left operand fits the policy width");
            let right = if matches!(scenario, Scenario::Edge)
                && matches!(operation, Operation::Div | Operation::Rem)
            {
                ArbiUint::zero_with_precision(width)
            } else {
                ArbiUint::with_precision_checked(
                    arbi_uint(right_bits, 1_337_u32.wrapping_add(index)),
                    width,
                )
                .expect("the generated right operand fits the policy width")
            };
            order_sub_operands(left, right, operation, scenario)
        })
        .collect()
}

/// Generates numerically identical Rug operands for one policy scenario.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_pairs(bits: usize, operation: Operation, scenario: Scenario) -> Vec<(Integer, Integer)> {
    let (left_bits, right_bits) = operand_widths(bits, operation, scenario);

    (0..SAMPLES)
        .map(|index| {
            let left = rug_uint(left_bits, 42_u32.wrapping_add(index));
            let right = if matches!(scenario, Scenario::Edge)
                && matches!(operation, Operation::Div | Operation::Rem)
            {
                Integer::new()
            } else {
                rug_uint(right_bits, 1_337_u32.wrapping_add(index))
            };
            order_sub_operands(left, right, operation, scenario)
        })
        .collect()
}

/// Converts the benchmark width to Rug's bit-count type.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_width(bits: usize) -> u32 {
    u32::try_from(bits).expect("policy benchmark widths fit in u32")
}

/// Returns the unsigned maximum representable by `bits` bits in Rug.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_max(bits: usize) -> Integer {
    Integer::from(-1).keep_bits(rug_width(bits))
}

/// Asserts that an Arbi result and a Rug policy result have the same value.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_value(actual: &ArbiUint, expected: &Integer) {
    assert_eq!(
        actual.to_string_radix(16),
        expected.to_string_radix(16),
        "Arbi and Rug policy results differ"
    );
}

/// Asserts equal optional outcomes and values.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_option(actual_result: Option<ArbiUint>, expected_result: Option<Integer>) {
    assert_eq!(
        actual_result.is_some(),
        expected_result.is_some(),
        "Arbi and Rug policy availability differs"
    );
    if let (Some(actual_value), Some(expected_value)) = (actual_result, expected_result) {
        verify_value(&actual_value, &expected_value);
    }
}

/// Asserts equal result status and successful values.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_result<E: core::fmt::Debug + PartialEq>(
    actual_result: Result<ArbiUint, E>,
    expected_result: Result<Integer, E>,
) {
    assert_eq!(
        actual_result.is_ok(),
        expected_result.is_ok(),
        "Arbi and Rug policy status differs"
    );
    match (actual_result, expected_result) {
        (Ok(actual_value), Ok(expected_value)) => verify_value(&actual_value, &expected_value),
        (Err(actual_error), Err(expected_error)) => {
            assert_eq!(actual_error, expected_error, "policy errors differ");
        }
        (Ok(_), Err(_)) | (Err(_), Ok(_)) => {}
    }
}

/// Asserts equal overflowing values and flags.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub fn verify_overflowing(actual: &(ArbiUint, bool), expected: &(Integer, bool)) {
    verify_value(&actual.0, &expected.0);
    assert_eq!(actual.1, expected.1, "overflow flags differ");
}

const fn operand_widths(bits: usize, operation: Operation, scenario: Scenario) -> (usize, usize) {
    match (operation, scenario) {
        // Two (bits - 4)-bit values sum to less than 2^(bits - 3). Keeping the
        // width nibble-aligned also preserves the exact-width hex generator.
        (Operation::Add, Scenario::Success) => {
            let operand_bits = bits.saturating_sub(4);
            (operand_bits, operand_bits)
        }
        // Two half-width values multiply to less than 2^bits for the even
        // benchmark widths used here.
        (Operation::Mul, Scenario::Success) => {
            let operand_bits = bits.checked_div(2).expect("two is non-zero");
            (operand_bits, operand_bits)
        }
        // A full-width dividend and half-width divisor exercise real division.
        (Operation::Div | Operation::Rem, Scenario::Success) => {
            (bits, bits.checked_div(2).expect("two is non-zero"))
        }
        // Exact-width addends and factors necessarily exceed the same bound.
        (Operation::Add | Operation::Mul, Scenario::Edge)
        | (Operation::Sub, Scenario::Success | Scenario::Edge) => (bits, bits),
        // The zero divisor is built separately, but retaining a non-zero width
        // here keeps both generators' shape calculations total.
        (Operation::Div | Operation::Rem, Scenario::Edge) => (bits, 1),
    }
}

fn order_sub_operands<T: Ord>(
    left: T,
    right: T,
    operation: Operation,
    scenario: Scenario,
) -> (T, T) {
    match (operation, scenario) {
        (Operation::Sub, Scenario::Success) if left < right => (right, left),
        (Operation::Sub, Scenario::Edge) => {
            assert!(left != right, "subtraction edge operands must differ");
            if left > right {
                (right, left)
            } else {
                (left, right)
            }
        }
        _ => (left, right),
    }
}
