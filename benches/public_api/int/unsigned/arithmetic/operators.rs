//! `Add`, `Sub`, `Mul` and their destination-reusing `assign_*` forms.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Add, Mul, Sub};

use divan::black_box;
use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Assign, Integer};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
use crate::int::{
    ladders::{ADDITIVE, MULTIPLICATIVE},
    support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, mp_uint_pairs},
};

/// `a + b`, allocating a fresh result.
mod add {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Add::add(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Add::add(black_box(left), black_box(right))));
            }
        });
    }
}

/// `a - b`, allocating a fresh result.
mod sub {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let (large, small) = if left >= right {
                    (left, right)
                } else {
                    (right, left)
                };
                let _output = black_box(Sub::sub(black_box(large), black_box(small)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let (large, small) = if left >= right {
                    (left, right)
                } else {
                    (right, left)
                };
                let _output =
                    black_box(Integer::from(Sub::sub(black_box(large), black_box(small))));
            }
        });
    }
}

/// `a * b` across the whole sub-quadratic threshold ladder.
mod mul {
    use super::*;

    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Mul::mul(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(Mul::mul(black_box(left), black_box(right))));
            }
        });
    }
}

/// `dst = a + b` into a destination reserved once, outside the timed region.
///
/// That reservation is the entire point of the API: the operator above has no
/// buffer to write into and allocates on every call. GMP's counterpart is
/// `Integer::assign`, which reuses its own allocation the same way.
mod assign_add {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        let mut result = MpUint::zero();
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
        let inputs = rug_uint_pairs(bits);
        let mut result = Integer::with_capacity(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign(Add::add(black_box(left), black_box(right)));
                let _output = black_box(&result);
            }
        });
    }
}

/// `dst = a - b` into a pre-reserved destination.
mod assign_sub {
    use super::*;

    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        let mut result = MpUint::zero();
        result.reserve(bits.div_ceil(64).saturating_add(1));
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let (large, small) = if left >= right {
                    (left, right)
                } else {
                    (right, left)
                };
                let _underflow = result.assign_sub(black_box(large), black_box(small));
                let _output = black_box(&result);
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = ADDITIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        let mut result = Integer::with_capacity(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let (large, small) = if left >= right {
                    (left, right)
                } else {
                    (right, left)
                };
                result.assign(Sub::sub(black_box(large), black_box(small)));
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
        let inputs = mp_uint_pairs(bits);
        let mut result = MpUint::zero();
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
        let inputs = rug_uint_pairs(bits);
        let mut result = Integer::with_capacity(bits.saturating_mul(2));
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                result.assign(Mul::mul(black_box(left), black_box(right)));
                let _output = black_box(&result);
            }
        });
    }
}

/// `dst = a * a` into a pre-reserved destination.
mod assign_square {
    use super::*;

    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        let mut result = MpUint::zero();
        result.reserve(bits.div_ceil(64).saturating_mul(2).saturating_add(1));
        bencher.bench_local(|| {
            for (value, _unused) in &inputs {
                result.assign_square(black_box(value));
                let _output = black_box(&result);
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MULTIPLICATIVE, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        let mut result = Integer::with_capacity(bits.saturating_mul(2));
        bencher.bench_local(|| {
            for (value, _unused) in &inputs {
                result.assign(black_box(value).square_ref());
                let _output = black_box(&result);
            }
        });
    }
}
