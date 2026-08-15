//! Truncating division on equal-width operands.
//!
//! Equal widths mean a one-limb quotient, so these cells are dominated by
//! normalisation, scratch acquisition and the final comparison rather than by
//! the quotient loop. That is the common shape in practice and the one where a
//! fixed per-call overhead shows up undiluted; [`super::shapes`](super::shapes)
//! covers the widths where the loop itself dominates.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Div, Rem};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, mp_uint_pairs, verify_mp_uint_division_pairs},
};

/// Quotient and remainder in one call.
mod div_rem {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem(black_box(right)));
            }
        });
    }

    /// `div_rem` consumes its operands, so the reference-taking `div_rem_ref`
    /// is used to keep operand clones out of the timed region.
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_ref(black_box(right)),
                ));
            }
        });
    }
}

/// Quotient only.
mod div_trunc {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_trunc(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Div::div(black_box(left), black_box(right))));
            }
        });
    }
}

/// Remainder only.
mod rem_trunc {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        verify_mp_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).rem_trunc(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Rem::rem(black_box(left), black_box(right))));
            }
        });
    }
}
