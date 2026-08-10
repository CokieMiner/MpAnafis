//! `wrapping_add`, `wrapping_sub`, `wrapping_mul`, `wrapping_div`, and
//! `wrapping_rem`.
//!
//! Rug emulates the bounded policy with GMP's remainder modulo `2^bits`.
//! Division and remainder need no truncation after a valid operation; their
//! edge cells return zero for a zero divisor, exactly like `ArbiUint`.

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
use super::cases::{rug_pairs, rug_width, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! wrapping_benches {
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

wrapping_benches!(
    add_success,
    add_overflow,
    Operation::Add,
    wrapping_add,
    rug_wrapping_add
);
wrapping_benches!(
    sub_success,
    sub_underflow,
    Operation::Sub,
    wrapping_sub,
    rug_wrapping_sub
);
wrapping_benches!(
    mul_success,
    mul_overflow,
    Operation::Mul,
    wrapping_mul,
    rug_wrapping_mul
);
wrapping_benches!(
    div_success,
    div_zero,
    Operation::Div,
    wrapping_div,
    rug_wrapping_div
);
wrapping_benches!(
    rem_success,
    rem_zero,
    Operation::Rem,
    wrapping_rem,
    rug_wrapping_rem
);

fn verify(operation: Operation, scenario: Scenario, bits: usize) {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        let arbi = arbi_pairs(bits, operation, scenario);
        let rug = rug_pairs(bits, operation, scenario);
        let width = rug_width(bits);

        for ((arbi_left, arbi_right), (rug_left, rug_right)) in arbi.iter().zip(&rug) {
            verify_value(arbi_left, rug_left);
            verify_value(arbi_right, rug_right);
            let (actual, expected) = match operation {
                Operation::Add => (
                    arbi_left.wrapping_add(arbi_right),
                    rug_wrapping_add(rug_left, rug_right, width),
                ),
                Operation::Sub => (
                    arbi_left.wrapping_sub(arbi_right),
                    rug_wrapping_sub(rug_left, rug_right, width),
                ),
                Operation::Mul => (
                    arbi_left.wrapping_mul(arbi_right),
                    rug_wrapping_mul(rug_left, rug_right, width),
                ),
                Operation::Div => (
                    arbi_left.wrapping_div(arbi_right),
                    rug_wrapping_div(rug_left, rug_right, width),
                ),
                Operation::Rem => (
                    arbi_left.wrapping_rem(arbi_right),
                    rug_wrapping_rem(rug_left, rug_right, width),
                ),
            };
            verify_value(&actual, &expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, scenario, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_wrapping_add(left: &Integer, right: &Integer, width: u32) -> Integer {
    Integer::from(Add::add(left, right)).keep_bits(width)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_wrapping_sub(left: &Integer, right: &Integer, width: u32) -> Integer {
    Integer::from(Sub::sub(left, right)).keep_bits(width)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_wrapping_mul(left: &Integer, right: &Integer, width: u32) -> Integer {
    Integer::from(Mul::mul(left, right)).keep_bits(width)
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_wrapping_div(left: &Integer, right: &Integer, _width: u32) -> Integer {
    if right.is_zero() {
        Integer::new()
    } else {
        Integer::from(Div::div(left, right))
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_wrapping_rem(left: &Integer, right: &Integer, _width: u32) -> Integer {
    if right.is_zero() {
        Integer::new()
    } else {
        Integer::from(Rem::rem(left, right))
    }
}
