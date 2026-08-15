//! Signed arithmetic, on the sign combinations that change the work done.
//!
//! Addition of like signs adds magnitudes; addition of unlike signs subtracts
//! them, compares first to find which is larger, and may flip the result sign.
//! Both are benchmarked, because only the second exercises the sign layer.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Add, Mul, Sub};

use divan::black_box;
use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Assign, Integer, ops::Pow};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, rug_int_pairs};
use crate::int::{
    ladders::{ADDITIVE, MULTIPLICATIVE, NARROW},
    support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_WIDE, mp_int, mp_int_pairs},
};

/// `a + b` with both operands positive: the magnitudes add.
mod add_like_signs {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Add::add(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Add::add(black_box(left), black_box(right))));
            }
        });
    }
}

/// `a + b` with opposite signs: the magnitudes subtract, after a comparison.
mod add_unlike_signs {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Add::add(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Add::add(black_box(left), black_box(right))));
            }
        });
    }
}

/// `a - b` with opposite signs, which adds magnitudes.
mod sub_unlike_signs {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Sub::sub(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Sub::sub(black_box(left), black_box(right))));
            }
        });
    }
}

/// `a * b` with one negative operand: the sign is one exclusive-or, so this
/// should track the unsigned product exactly.
mod mul {
    use super::*;

    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Mul::mul(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Mul::mul(black_box(left), black_box(right))));
            }
        });
    }
}

/// `dst = a + b` into a pre-reserved destination.
mod assign_add {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, false);
        let mut result = MpInt::zero();
        result.reserve(bits.div_ceil(64).saturating_add(1));
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign_add(black_box(left), black_box(right));
                let _output = black_box(&result);
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, false);
        let mut result = Integer::with_capacity(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign(Add::add(black_box(left), black_box(right)));
                let _output = black_box(&result);
            }
        });
    }
}

/// `dst = a * b` into a destination reserved for the full product width.
mod assign_mul {
    use super::*;

    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, false, true);
        let mut result = MpInt::zero();
        result.reserve(bits.div_ceil(64).saturating_mul(2).saturating_add(1));
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign_mul(black_box(left), black_box(right));
                let _output = black_box(&result);
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, false, true);
        let mut result = Integer::with_capacity(bits.saturating_mul(2));
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign(Mul::mul(black_box(left), black_box(right)));
                let _output = black_box(&result);
            }
        });
    }
}

/// `a + b * c` with a negative addend.
mod mul_add {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let addend = mp_int(bits, 42, true);
        let multiplier = mp_int(bits, 1_337, false);
        let multiplicand = mp_int(bits, 9_999, false);
        bencher.bench_local(|| {
            let _output = black_box(
                black_box(&addend).mul_add(black_box(&multiplier), black_box(&multiplicand)),
            );
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let addend = rug_int(bits, 42, true);
        let multiplier = rug_int(bits, 1_337, false);
        let multiplicand = rug_int(bits, 9_999, false);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Add::add(
                black_box(&addend),
                Mul::mul(black_box(&multiplier), black_box(&multiplicand)),
            )));
        });
    }
}

/// `a * a` from a negative operand, where the sign is known to cancel.
mod square {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).square());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).square_ref()));
        });
    }
}

/// An odd exponent over a negative base, so the result stays negative.
mod pow {
    use super::*;

    const EXPONENT: u32 = 17;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let base = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&base).pow(EXPONENT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let base = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Pow::pow(black_box(&base), EXPONENT)));
        });
    }
}
