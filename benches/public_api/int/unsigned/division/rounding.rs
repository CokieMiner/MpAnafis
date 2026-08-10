//! The Euclidean, floor and ceiling division families.
//!
//! On unsigned operands every one of these agrees with truncation, so the cell
//! measures what the rounding wrapper costs when it has nothing to correct.
//! GMP names the same three roundings `mpz_fdiv_*` (floor), `mpz_cdiv_*`
//! (ceiling) and `mpz_tdiv_*` (truncate), which Rug exposes as `div_floor`,
//! `div_ceil` and the plain operators.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Div, Rem};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{
    Integer,
    ops::{DivRounding, RemRounding},
};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{
        SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, arbi_uint_pairs, verify_arbi_uint_division_pairs,
    },
};

mod div_euclid {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_euclid(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(DivRounding::div_euc(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod rem_euclid {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).rem_euclid(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(RemRounding::rem_euc(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod div_floor {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_floor(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(DivRounding::div_floor(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod mod_floor {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).mod_floor(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(RemRounding::rem_floor(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod div_ceil {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_ceil(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(DivRounding::div_ceil(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

/// Floor quotient and remainder in one call.
mod div_rem_floor {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem_floor(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_floor_ref(black_box(right)),
                ));
            }
        });
    }
}

/// Euclidean quotient and remainder in one call.
mod div_rem_euclid {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem_euclid(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_euc_ref(black_box(right)),
                ));
            }
        });
    }
}

/// The `checked_` forms of the same roundings, on in-domain operands.
mod checked_rounding {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        verify_arbi_uint_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _floor = black_box(black_box(left).checked_div_floor(black_box(right)));
                let _ceil = black_box(black_box(left).checked_div_ceil(black_box(right)));
                let _euclid = black_box(black_box(left).checked_div_euclid(black_box(right)));
                let _trunc = black_box(black_box(left).checked_div_trunc(black_box(right)));
                let _modulo = black_box(black_box(left).checked_mod_floor(black_box(right)));
                let _rem_euclid = black_box(black_box(left).checked_rem_euclid(black_box(right)));
                let _rem_trunc = black_box(black_box(left).checked_rem_trunc(black_box(right)));
            }
        });
    }

    /// Seven GMP divisions, one per `checked_` method on the `arbi` side.
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _floor = black_box(Integer::from(DivRounding::div_floor(
                    black_box(left),
                    right,
                )));
                let _ceil = black_box(Integer::from(DivRounding::div_ceil(black_box(left), right)));
                let _euclid =
                    black_box(Integer::from(DivRounding::div_euc(black_box(left), right)));
                let _trunc = black_box(Integer::from(Div::div(black_box(left), right)));
                let _modulo = black_box(Integer::from(RemRounding::rem_floor(
                    black_box(left),
                    right,
                )));
                let _rem_euclid =
                    black_box(Integer::from(RemRounding::rem_euc(black_box(left), right)));
                let _rem_trunc = black_box(Integer::from(Rem::rem(black_box(left), right)));
            }
        });
    }
}
