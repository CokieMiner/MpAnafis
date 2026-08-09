//! `saturating_add`, `saturating_sub`, `saturating_mul`, `saturating_div`, and
//! `saturating_rem`.
//!
//! Every operand is bounded. Rug computes the full GMP result and applies the
//! same clamp: the `2^bits - 1` maximum for add/mul overflow, zero for unsigned
//! subtraction underflow, and zero for a zero divisor. Thus the Rug edge cells
//! no longer report an unclamped, mathematically different result.

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
use super::cases::{rug_max, rug_pairs, rug_width, verify_value};
use crate::int::{ladders::NARROW, support::SAMPLE_SIZE_FAST};

macro_rules! saturating_benches {
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
                let max = rug_max(bits);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output =
                            black_box($rug(black_box(left), black_box(right), width, &max));
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
                let max = rug_max(bits);
                bencher.bench_local(|| {
                    for (left, right) in &inputs {
                        let _output =
                            black_box($rug(black_box(left), black_box(right), width, &max));
                    }
                });
            }
        }
    };
}

saturating_benches!(
    add_success,
    add_overflow,
    Operation::Add,
    saturating_add,
    rug_saturating_add
);
saturating_benches!(
    sub_success,
    sub_underflow,
    Operation::Sub,
    saturating_sub,
    rug_saturating_sub
);
saturating_benches!(
    mul_success,
    mul_overflow,
    Operation::Mul,
    saturating_mul,
    rug_saturating_mul
);
saturating_benches!(
    div_success,
    div_zero,
    Operation::Div,
    saturating_div,
    rug_saturating_div
);
saturating_benches!(
    rem_success,
    rem_zero,
    Operation::Rem,
    saturating_rem,
    rug_saturating_rem
);

fn verify(operation: Operation, scenario: Scenario, bits: usize) {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        let mp = mp_pairs(bits, operation, scenario);
        let rug = rug_pairs(bits, operation, scenario);
        let width = rug_width(bits);
        let max = rug_max(bits);

        for ((mp_left, mp_right), (rug_left, rug_right)) in mp.iter().zip(&rug) {
            verify_value(mp_left, rug_left);
            verify_value(mp_right, rug_right);
            let (actual, expected) = match operation {
                Operation::Add => (
                    mp_left.saturating_add(mp_right),
                    rug_saturating_add(rug_left, rug_right, width, &max),
                ),
                Operation::Sub => (
                    mp_left.saturating_sub(mp_right),
                    rug_saturating_sub(rug_left, rug_right, width, &max),
                ),
                Operation::Mul => (
                    mp_left.saturating_mul(mp_right),
                    rug_saturating_mul(rug_left, rug_right, width, &max),
                ),
                Operation::Div => (
                    mp_left.saturating_div(mp_right),
                    rug_saturating_div(rug_left, rug_right, width, &max),
                ),
                Operation::Rem => (
                    mp_left.saturating_rem(mp_right),
                    rug_saturating_rem(rug_left, rug_right, width, &max),
                ),
            };
            verify_value(&actual, &expected);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
    let _ = (operation, scenario, bits);
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_saturating_add(left: &Integer, right: &Integer, _width: u32, max: &Integer) -> Integer {
    let result = Integer::from(Add::add(left, right));
    if result > *max { max.clone() } else { result }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_saturating_sub(left: &Integer, right: &Integer, _width: u32, _max: &Integer) -> Integer {
    let result = Integer::from(Sub::sub(left, right));
    if result.is_negative() {
        Integer::new()
    } else {
        result
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_saturating_mul(left: &Integer, right: &Integer, _width: u32, max: &Integer) -> Integer {
    let result = Integer::from(Mul::mul(left, right));
    if result > *max { max.clone() } else { result }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_saturating_div(left: &Integer, right: &Integer, _width: u32, _max: &Integer) -> Integer {
    if right.is_zero() {
        Integer::new()
    } else {
        Integer::from(Div::div(left, right))
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
fn rug_saturating_rem(left: &Integer, right: &Integer, _width: u32, _max: &Integer) -> Integer {
    if right.is_zero() {
        Integer::new()
    } else {
        Integer::from(Rem::rem(left, right))
    }
}
