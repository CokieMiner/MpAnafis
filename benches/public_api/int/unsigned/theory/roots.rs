//! Integer roots and the perfect-square test.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_square_plus_one, rug_true_squares, rug_uint};
use crate::int::{
    ladders::{NARROW, ROOTS},
    support::{
        SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_WIDE, mp_square_plus_one, mp_true_squares,
        mp_uint,
    },
};

mod isqrt {
    use super::*;

    #[divan::bench(args = ROOTS, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).isqrt());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ROOTS, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).sqrt_ref()));
        });
    }
}

mod sqrt_rem {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).sqrt_rem());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let remainder = Integer::new();
            let _output = black_box(black_box(&value).clone().sqrt_rem(remainder));
        });
    }
}

/// The cube root, standing in for the general `nth_root`.
mod nth_root {
    use super::*;

    const DEGREE: u32 = 3;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).nth_root(DEGREE));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).root_ref(DEGREE)));
        });
    }
}

/// The test on a random value, which the residue screen rejects immediately.
mod is_perfect_square_random {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_perfect_square());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_perfect_square());
        });
    }
}

/// The test on an exact square, where the full root must be computed.
mod is_perfect_square_true {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let values = mp_true_squares(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_perfect_square());
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let values = rug_true_squares(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_perfect_square());
            }
        });
    }
}

/// The near miss: passes the cheap screens, fails on the root.
mod is_perfect_square_plus_one {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let values = mp_square_plus_one(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_perfect_square());
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let values = rug_square_plus_one(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_perfect_square());
            }
        });
    }
}
