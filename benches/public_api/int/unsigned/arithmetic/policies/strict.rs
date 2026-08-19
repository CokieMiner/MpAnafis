//! `strict_add`, `strict_sub`, `strict_mul`, `strict_div`, and `strict_rem`.
//!
//! These cells use bounded, in-domain operands and include the successful
//! panic-contract check. Exceptional calls are not timed: catching an unwind
//! would measure the panic runtime and hook rather than integer arithmetic.
//! The corresponding failures are covered by the checked and `Result` edge
//! cells and by the API tests.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::{Add, Div, Mul, Rem, Sub};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::cases::{Operation, Scenario, mp_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::cases::{rug_pairs, rug_width, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! strict_benches {
    ($module:ident, $operation:expr, $method:ident, $rug:path) => {
        mod $module {
            use super::*;

            #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                verify($operation, bits);
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
                verify($operation, bits);
                let inputs = rug_pairs(bits, $operation, Scenario::Success);
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

strict_benches!(add_success, Operation::Add, strict_add, rug_strict_add);
strict_benches!(sub_success, Operation::Sub, strict_sub, rug_strict_sub);
strict_benches!(mul_success, Operation::Mul, strict_mul, rug_strict_mul);
strict_benches!(div_success, Operation::Div, strict_div, rug_strict_div);
strict_benches!(rem_success, Operation::Rem, strict_rem, rug_strict_rem);

fn verify(operation: Operation, bits: usize) {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        let mp = mp_pairs(bits, operation, Scenario::Success);
        let rug = rug_pairs(bits, operation, Scenario::Success);
        let width = rug_width(bits);

        for ((mp_left, mp_right), (rug_left, rug_right)) in mp.iter().zip(&rug) {
            verify_value(mp_left, rug_left);
            verify_value(mp_right, rug_right);
            let (actual, expected) = match operation {
                Operation::Add => (
                    mp_left.strict_add(mp_right),
                    rug_strict_add(rug_left, rug_right, width),
                ),
                Operation::Sub => (
                    mp_left.strict_sub(mp_right),
                    rug_strict_sub(rug_left, rug_right, width),
                ),
                Operation::Mul => (
                    mp_left.strict_mul(mp_right),
                    rug_strict_mul(rug_left, rug_right, width),
                ),
                Operation::Div => (
                    mp_left.strict_div(mp_right),
                    rug_strict_div(rug_left, rug_right, width),
                ),
                Operation::Rem => (
                    mp_left.strict_rem(mp_right),
                    rug_strict_rem(rug_left, rug_right, width),
                ),
            };
            verify_value(&actual, &expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_strict_add(left: &Integer, right: &Integer, width: u32) -> Integer {
    let result = Integer::from(Add::add(left, right));
    assert!(
        result.significant_bits() <= width,
        "strict addition overflow"
    );
    result
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_strict_sub(left: &Integer, right: &Integer, _width: u32) -> Integer {
    let result = Integer::from(Sub::sub(left, right));
    assert!(!result.is_negative(), "strict subtraction underflow");
    result
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_strict_mul(left: &Integer, right: &Integer, width: u32) -> Integer {
    let result = Integer::from(Mul::mul(left, right));
    assert!(
        result.significant_bits() <= width,
        "strict multiplication overflow"
    );
    result
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_strict_div(left: &Integer, right: &Integer, _width: u32) -> Integer {
    assert!(!right.is_zero(), "strict division by zero");
    Integer::from(Div::div(left, right))
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_strict_rem(left: &Integer, right: &Integer, _width: u32) -> Integer {
    assert!(!right.is_zero(), "strict remainder by zero");
    Integer::from(Rem::rem(left, right))
}
