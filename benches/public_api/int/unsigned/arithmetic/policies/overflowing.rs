//! `overflowing_add`, `overflowing_sub`, `overflowing_mul`, `overflowing_div`,
//! and `overflowing_rem`.
//!
//! Both implementations return the bounded result plus an equivalent status
//! flag. The edge cells force that flag to `true`; every success cell proves it
//! remains `false`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use core::ops::{Add, Div, Mul, Rem, Sub};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::cases::{EDGE_WIDTH, Operation, Scenario, arbi_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::cases::{rug_pairs, rug_width, verify_overflowing, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! overflowing_benches {
    ($success:ident, $edge:ident, $operation:expr, $method:ident, $rug:path) => {
        mod $success {
            use super::*;

            #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
            fn arbi(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Success, bits);
                let inputs = arbi_pairs(bits, $operation, Scenario::Success);
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
            fn arbi(bencher: divan::Bencher, bits: usize) {
                verify($operation, Scenario::Edge, bits);
                let inputs = arbi_pairs(bits, $operation, Scenario::Edge);
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

overflowing_benches!(
    add_success,
    add_overflow,
    Operation::Add,
    overflowing_add,
    rug_overflowing_add
);
overflowing_benches!(
    sub_success,
    sub_underflow,
    Operation::Sub,
    overflowing_sub,
    rug_overflowing_sub
);
overflowing_benches!(
    mul_success,
    mul_overflow,
    Operation::Mul,
    overflowing_mul,
    rug_overflowing_mul
);
overflowing_benches!(
    div_success,
    div_zero,
    Operation::Div,
    overflowing_div,
    rug_overflowing_div
);
overflowing_benches!(
    rem_success,
    rem_zero,
    Operation::Rem,
    overflowing_rem,
    rug_overflowing_rem
);

fn verify(operation: Operation, scenario: Scenario, bits: usize) {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        let arbi = arbi_pairs(bits, operation, scenario);
        let rug = rug_pairs(bits, operation, scenario);
        let width = rug_width(bits);
        let expected_flag = matches!(scenario, Scenario::Edge);

        for ((arbi_left, arbi_right), (rug_left, rug_right)) in arbi.iter().zip(&rug) {
            verify_value(arbi_left, rug_left);
            verify_value(arbi_right, rug_right);
            let (actual, expected) = match operation {
                Operation::Add => (
                    arbi_left.overflowing_add(arbi_right),
                    rug_overflowing_add(rug_left, rug_right, width),
                ),
                Operation::Sub => (
                    arbi_left.overflowing_sub(arbi_right),
                    rug_overflowing_sub(rug_left, rug_right, width),
                ),
                Operation::Mul => (
                    arbi_left.overflowing_mul(arbi_right),
                    rug_overflowing_mul(rug_left, rug_right, width),
                ),
                Operation::Div => (
                    arbi_left.overflowing_div(arbi_right),
                    rug_overflowing_div(rug_left, rug_right, width),
                ),
                Operation::Rem => (
                    arbi_left.overflowing_rem(arbi_right),
                    rug_overflowing_rem(rug_left, rug_right, width),
                ),
            };
            assert_eq!(
                actual.1, expected_flag,
                "overflowing benchmark took the wrong policy path"
            );
            verify_overflowing(&actual, &expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, scenario, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_overflowing_add(left: &Integer, right: &Integer, width: u32) -> (Integer, bool) {
    let result = Integer::from(Add::add(left, right));
    let overflowed = result.significant_bits() > width;
    (result.keep_bits(width), overflowed)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_overflowing_sub(left: &Integer, right: &Integer, width: u32) -> (Integer, bool) {
    let result = Integer::from(Sub::sub(left, right));
    let underflowed = result.is_negative();
    (result.keep_bits(width), underflowed)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_overflowing_mul(left: &Integer, right: &Integer, width: u32) -> (Integer, bool) {
    let result = Integer::from(Mul::mul(left, right));
    let overflowed = result.significant_bits() > width;
    (result.keep_bits(width), overflowed)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_overflowing_div(left: &Integer, right: &Integer, _width: u32) -> (Integer, bool) {
    if right.is_zero() {
        (Integer::new(), true)
    } else {
        (Integer::from(Div::div(left, right)), false)
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_overflowing_rem(left: &Integer, right: &Integer, _width: u32) -> (Integer, bool) {
    if right.is_zero() {
        (Integer::new(), true)
    } else {
        (Integer::from(Rem::rem(left, right)), false)
    }
}
