//! `div_rem` on the operand shapes that reach the recursive divider.
//!
//! The dividend-to-divisor ratio, not the width, decides which algorithm runs:
//! `2n / n` is the balanced recursive case, `3n/2 / n` takes the uneven split,
//! and an equal-limb pair with a quotient of at least two is the boundary where
//! the schoolbook path is still chosen.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_div_pairs_2n_n, rug_div_pairs_3n2_n, rug_div_pairs_same_limbs_ge_2};
use crate::int::{
    ladders::DIVISION,
    support::{
        SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, mp_div_pairs_2n_n, mp_div_pairs_3n2_n,
        mp_div_pairs_same_limbs_ge_2,
    },
};

mod div_rem_2n_by_n {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_div_pairs_2n_n(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_div_pairs_2n_n(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_ref(black_box(right)),
                ));
            }
        });
    }
}

mod div_rem_3n2_by_n {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_div_pairs_3n2_n(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_div_pairs_3n2_n(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_ref(black_box(right)),
                ));
            }
        });
    }
}

mod div_rem_same_limbs_ge_2 {
    use super::*;

    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_div_pairs_same_limbs_ge_2(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).div_rem(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DIVISION, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_div_pairs_same_limbs_ge_2(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(<(Integer, Integer)>::from(
                    black_box(left).div_rem_ref(black_box(right)),
                ));
            }
        });
    }
}
