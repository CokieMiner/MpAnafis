//! `checked_add`, `checked_sub`, `checked_mul`, `checked_div`, and `checked_rem`.
//!
//! Successful cells return `Some`. Edge cells return `None` after bounded
//! overflow or underflow, or immediately for a zero divisor.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::{Add, Div, Mul, Rem, Sub};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::cases::{EDGE_WIDTH, Operation, Scenario, mp_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::cases::{rug_pairs, rug_width, verify_option, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! checked_benches {
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

checked_benches!(
    add_success,
    add_overflow,
    Operation::Add,
    checked_add,
    rug_checked_add
);
checked_benches!(
    sub_success,
    sub_underflow,
    Operation::Sub,
    checked_sub,
    rug_checked_sub
);
checked_benches!(
    mul_success,
    mul_overflow,
    Operation::Mul,
    checked_mul,
    rug_checked_mul
);
checked_benches!(
    div_success,
    div_zero,
    Operation::Div,
    checked_div,
    rug_checked_div
);
checked_benches!(
    rem_success,
    rem_zero,
    Operation::Rem,
    checked_rem,
    rug_checked_rem
);

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
                    mp_left.checked_add(mp_right),
                    rug_checked_add(rug_left, rug_right, width),
                ),
                Operation::Sub => (
                    mp_left.checked_sub(mp_right),
                    rug_checked_sub(rug_left, rug_right, width),
                ),
                Operation::Mul => (
                    mp_left.checked_mul(mp_right),
                    rug_checked_mul(rug_left, rug_right, width),
                ),
                Operation::Div => (
                    mp_left.checked_div(mp_right),
                    rug_checked_div(rug_left, rug_right, width),
                ),
                Operation::Rem => (
                    mp_left.checked_rem(mp_right),
                    rug_checked_rem(rug_left, rug_right, width),
                ),
            };
            assert_eq!(
                actual.is_some(),
                expect_success,
                "checked benchmark took the wrong policy path"
            );
            verify_option(actual, expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, scenario, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_checked_add(left: &Integer, right: &Integer, width: u32) -> Option<Integer> {
    let result = Integer::from(Add::add(left, right));
    (result.significant_bits() <= width).then_some(result)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_checked_sub(left: &Integer, right: &Integer, _width: u32) -> Option<Integer> {
    let result = Integer::from(Sub::sub(left, right));
    (!result.is_negative()).then_some(result)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_checked_mul(left: &Integer, right: &Integer, width: u32) -> Option<Integer> {
    let result = Integer::from(Mul::mul(left, right));
    (result.significant_bits() <= width).then_some(result)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_checked_div(left: &Integer, right: &Integer, _width: u32) -> Option<Integer> {
    (!right.is_zero()).then(|| Integer::from(Div::div(left, right)))
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_checked_rem(left: &Integer, right: &Integer, _width: u32) -> Option<Integer> {
    (!right.is_zero()).then(|| Integer::from(Rem::rem(left, right)))
}
