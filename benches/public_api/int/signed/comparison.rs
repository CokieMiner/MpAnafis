//! Signed ordering, equality and hashing.
//!
//! Opposite signs make the comparison trivial, so the operands here share a
//! sign: the answer then depends on the magnitudes and the comparison has to do
//! the same work its unsigned counterpart does, plus the sign dispatch.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::hash::{Hash, Hasher};

use divan::black_box;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, rug_int_pairs};
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_int, mp_int_pairs},
};

/// Ordering of two negative values, where the magnitude order is reversed.
mod cmp {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).cmp(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).cmp(black_box(right)));
            }
        });
    }
}

mod eq {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left) == black_box(right));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left) == black_box(right));
            }
        });
    }
}

mod hash {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            black_box(&value).hash(&mut hasher);
            let _output = black_box(hasher.finish());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            black_box(&value).hash(&mut hasher);
            let _output = black_box(hasher.finish());
        });
    }
}

mod min {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).min(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).min(black_box(right)).clone());
            }
        });
    }
}

mod max {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).max(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).max(black_box(right)).clone());
            }
        });
    }
}
