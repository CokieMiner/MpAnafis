//! Signed division: the one place the rounding families genuinely differ.
//!
//! With a negative dividend, truncation rounds toward zero, flooring rounds
//! down, ceiling rounds up and Euclidean division forces a non-negative
//! remainder — four different answers from the same operands, each needing its
//! own correction after the underlying magnitude division. On unsigned operands
//! all four agree and the corrections are dead code, which is why these cells
//! all use a negative dividend.

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
use crate::int::support::rug_int_pairs;
use crate::int::{
    ladders::DIVISION,
    support::{
        SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, arbi_int_pairs, verify_arbi_int_division_pairs,
    },
};

mod div_trunc {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_trunc(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Div::div(black_box(left), black_box(right))));
            }
        });
    }
}

mod rem_trunc {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).rem_trunc(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Rem::rem(black_box(left), black_box(right))));
            }
        });
    }
}

mod div_floor {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_floor(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
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
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).mod_floor(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
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
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_ceil(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
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

mod div_euclid {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_euclid(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
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
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).rem_euclid(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
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

mod div_rem {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_int_pairs(bits, true, false);
        verify_arbi_int_division_pairs(&inputs);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_ref(black_box(right)),
                ));
            }
        });
    }
}
