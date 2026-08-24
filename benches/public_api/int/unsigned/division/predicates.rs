//! Divisibility tests, which may answer without producing a quotient.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint_pairs},
};

/// `b | a` asked from the dividend.
mod is_divisible_by {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).is_divisible_by(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).is_divisible(black_box(right)));
            }
        });
    }
}

/// The same question asked from the divisor.
mod is_divisor_of {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(right).is_divisor_of(black_box(left)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).is_divisible(black_box(right)));
            }
        });
    }
}
