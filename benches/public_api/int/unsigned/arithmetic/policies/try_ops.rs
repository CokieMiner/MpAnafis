//! `try_add`, `try_sub`, `try_mul`, `try_div`, and `try_rem`.
//!
//! Named `try_ops` because `try` is a reserved word. Rug returns the same
//! `MpError` variant as the Mp method so both cells include equivalent
//! `Result` construction.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::{Add, Div, Mul, Rem, Sub};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use mp_anafis::MpError;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::cases::{EDGE_WIDTH, Operation, Scenario, mp_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::cases::{rug_pairs, rug_width, verify_result, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! try_benches {
    ($success:ident, $edge:ident, $operation:expr, $method:ident, $rug:path) => {
        mod $success {
            use super::*;

            #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Success, bits);
                let inputs = mp_pairs(bits, $operation, Scenario::Success);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output = black_box(black_box(left).$method(black_box(right)));
                    }
                });
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
            fn rug(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Success, bits);
                let inputs = rug_pairs(bits, $operation, Scenario::Success);
                let width = rug_width(bits);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output = black_box($rug(black_box(left), black_box(right), width));
                    }
                });
            }
        }

        mod $edge {
            use super::*;

            #[divan::bench(args = EDGE_WIDTH, sample_size = SAMPLE_SIZE_FAST)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Edge, bits);
                let inputs = mp_pairs(bits, $operation, Scenario::Edge);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output = black_box(black_box(left).$method(black_box(right)));
                    }
                });
            }

            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            #[divan::bench(args = EDGE_WIDTH, sample_size = SAMPLE_SIZE_FAST)]
            fn rug(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Edge, bits);
                let inputs = rug_pairs(bits, $operation, Scenario::Edge);
                let width = rug_width(bits);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output = black_box($rug(black_box(left), black_box(right), width));
                    }
                });
            }
        }
    };
}

try_benches!(
    add_success,
    add_overflow,
    Operation::Add,
    try_add,
    rug_try_add
);
try_benches!(
    sub_success,
    sub_underflow,
    Operation::Sub,
    try_sub,
    rug_try_sub
);
try_benches!(
    mul_success,
    mul_overflow,
    Operation::Mul,
    try_mul,
    rug_try_mul
);
try_benches!(div_success, div_zero, Operation::Div, try_div, rug_try_div);
try_benches!(rem_success, rem_zero, Operation::Rem, try_rem, rug_try_rem);

fn verify(operation: Operation, scenario: Scenario, bits: usize) {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        let mp = mp_pairs(bits, operation, scenario);
        let rug = rug_pairs(bits, operation, scenario);
        let width = rug_width(bits);
        let expect_success = matches!(scenario, Scenario::Success);

        for ((mp_left, mp_right), (rug_left, rug_right)) in mp.iter().zip(&rug) {
            verify_value(mp_left, rug_left);
            verify_value(mp_right, rug_right);
            let (actual, expected) = match operation {
                Operation::Add => (
                    mp_left.try_add(mp_right),
                    rug_try_add(rug_left, rug_right, width),
                ),
                Operation::Sub => (
                    mp_left.try_sub(mp_right),
                    rug_try_sub(rug_left, rug_right, width),
                ),
                Operation::Mul => (
                    mp_left.try_mul(mp_right),
                    rug_try_mul(rug_left, rug_right, width),
                ),
                Operation::Div => (
                    mp_left.try_div(mp_right),
                    rug_try_div(rug_left, rug_right, width),
                ),
                Operation::Rem => (
                    mp_left.try_rem(mp_right),
                    rug_try_rem(rug_left, rug_right, width),
                ),
            };
            assert_eq!(
                actual.is_ok(),
                expect_success,
                "try benchmark took the wrong policy path"
            );
            verify_result(actual, expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, scenario, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_try_add(left: &Integer, right: &Integer, width: u32) -> Result<Integer, MpError> {
    let result = Integer::from(Add::add(left, right));
    if result.significant_bits() > width {
        Err(MpError::Overflow)
    } else {
        Ok(result)
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_try_sub(left: &Integer, right: &Integer, _width: u32) -> Result<Integer, MpError> {
    let result = Integer::from(Sub::sub(left, right));
    if result.is_negative() {
        Err(MpError::Underflow)
    } else {
        Ok(result)
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_try_mul(left: &Integer, right: &Integer, width: u32) -> Result<Integer, MpError> {
    let result = Integer::from(Mul::mul(left, right));
    if result.significant_bits() > width {
        Err(MpError::Overflow)
    } else {
        Ok(result)
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_try_div(left: &Integer, right: &Integer, _width: u32) -> Result<Integer, MpError> {
    if right.is_zero() {
        Err(MpError::DivisionByZero)
    } else {
        Ok(Integer::from(Div::div(left, right)))
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_try_rem(left: &Integer, right: &Integer, _width: u32) -> Result<Integer, MpError> {
    if right.is_zero() {
        Err(MpError::DivisionByZero)
    } else {
        Ok(Integer::from(Rem::rem(left, right)))
    }
}
