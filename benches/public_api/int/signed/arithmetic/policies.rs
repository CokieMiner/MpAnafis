//! Signed wrapping and overflowing arithmetic with bounded operand shapes.
//!
//! GMP equivalents compose exact arithmetic with signed bit reduction; comparing
//! the exact and reduced values supplies the same overflow flag.

use mp_anafis::{BoundedPrecision, MpInt};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLES, mp_int, paired_bench},
};

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
}

mod wrapping_add {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired scenarios inherit the category's explicit imports and operand constructors."
    )]
    use super::*;

    paired_bench!(success, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Add, false) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.wrapping_add(b),
        rug: |bits| rug_pairs(bits, Operation::Add, false) =>
            |(a, b, width): &(Integer, Integer, u32)| Integer::from(a + b).keep_signed_bits(*width),
    );
    paired_bench!(overflow, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Add, true) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.wrapping_add(b),
        rug: |bits| rug_pairs(bits, Operation::Add, true) =>
            |(a, b, width): &(Integer, Integer, u32)| Integer::from(a + b).keep_signed_bits(*width),
    );
}

mod wrapping_mul {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired scenarios inherit the category's explicit imports and operand constructors."
    )]
    use super::*;

    paired_bench!(success, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Mul, false) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.wrapping_mul(b),
        rug: |bits| rug_pairs(bits, Operation::Mul, false) =>
            |(a, b, width): &(Integer, Integer, u32)| Integer::from(a * b).keep_signed_bits(*width),
    );
    paired_bench!(overflow, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Mul, true) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.wrapping_mul(b),
        rug: |bits| rug_pairs(bits, Operation::Mul, true) =>
            |(a, b, width): &(Integer, Integer, u32)| Integer::from(a * b).keep_signed_bits(*width),
    );
}

mod overflowing_add {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired scenarios inherit the category's explicit imports and operand constructors."
    )]
    use super::*;

    paired_bench!(success, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Add, false) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_add(b),
        rug: |bits| rug_pairs(bits, Operation::Add, false) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a + b), *width),
    );
    paired_bench!(overflow, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Add, true) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_add(b),
        rug: |bits| rug_pairs(bits, Operation::Add, true) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a + b), *width),
    );
}

mod overflowing_sub {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired scenarios inherit the category's explicit imports and operand constructors."
    )]
    use super::*;

    paired_bench!(success, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Sub, false) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_sub(b),
        rug: |bits| rug_pairs(bits, Operation::Sub, false) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a - b), *width),
    );
    paired_bench!(overflow, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Sub, true) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_sub(b),
        rug: |bits| rug_pairs(bits, Operation::Sub, true) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a - b), *width),
    );
}

mod overflowing_mul {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired scenarios inherit the category's explicit imports and operand constructors."
    )]
    use super::*;

    paired_bench!(success, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Mul, false) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_mul(b),
        rug: |bits| rug_pairs(bits, Operation::Mul, false) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a * b), *width),
    );
    paired_bench!(overflow, NARROW,
        mp: |bits| mp_pairs(bits, Operation::Mul, true) =>
            |(a, b, _): &(MpInt, MpInt, u32)| a.overflowing_mul(b),
        rug: |bits| rug_pairs(bits, Operation::Mul, true) =>
            |(a, b, width): &(Integer, Integer, u32)| rug_reduce(Integer::from(a * b), *width),
    );
}

fn mp_pairs(bits: usize, operation: Operation, overflow: bool) -> Vec<(MpInt, MpInt, u32)> {
    let width = BoundedPrecision::new(bits).expect("benchmark width is valid");
    let native_width = u32::try_from(bits).expect("benchmark width fits u32");
    let (operand_bits, left_negative, right_negative) = operand_shape(bits, operation, overflow);
    (0..SAMPLES)
        .map(|index| {
            let left = if overflow {
                if matches!(operation, Operation::Sub) {
                    MpInt::max_for_precision(bits)
                } else {
                    MpInt::min_for_precision(bits)
                }
            } else {
                MpInt::with_precision_checked(
                    mp_int(operand_bits, 42_u32.wrapping_add(index), left_negative),
                    width,
                )
                .expect("generated signed operand fits")
            };
            let right = MpInt::with_precision_checked(
                mp_int(operand_bits, 1_337_u32.wrapping_add(index), right_negative),
                width,
            )
            .expect("generated signed operand fits");
            (left, right, native_width)
        })
        .collect()
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_pairs(bits: usize, operation: Operation, overflow: bool) -> Vec<(Integer, Integer, u32)> {
    let width = u32::try_from(bits).expect("benchmark width fits u32");
    let (operand_bits, left_negative, right_negative) = operand_shape(bits, operation, overflow);
    (0..SAMPLES)
        .map(|index| {
            let left = if overflow {
                let magnitude_bits = width.saturating_sub(1);
                if matches!(operation, Operation::Sub) {
                    Integer::from(-1).keep_bits(magnitude_bits)
                } else {
                    Integer::from(-1) << magnitude_bits
                }
            } else {
                rug_int(operand_bits, 42_u32.wrapping_add(index), left_negative)
            };
            (
                left,
                rug_int(operand_bits, 1_337_u32.wrapping_add(index), right_negative),
                width,
            )
        })
        .collect()
}

const fn operand_shape(bits: usize, operation: Operation, overflow: bool) -> (usize, bool, bool) {
    let operand_bits = if overflow {
        bits.saturating_sub(4)
    } else if matches!(operation, Operation::Mul) {
        bits.div_euclid(2).saturating_sub(4)
    } else {
        bits.saturating_sub(8)
    };
    // Hexadecimal-aligned successful operands leave a sign-bit margin. The
    // overflowing generators replace the left operand by MIN (add/mul) or MAX
    // (sub); the negative right operand makes every exact result overflow.
    let left_negative = !matches!(operation, Operation::Sub);
    (operand_bits, left_negative, true)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_reduce(exact: Integer, width: u32) -> (Integer, bool) {
    let original = exact.clone();
    let wrapped = exact.keep_signed_bits(width);
    let overflow = wrapped != original;
    (wrapped, overflow)
}
