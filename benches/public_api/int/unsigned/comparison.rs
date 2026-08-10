//! Ordering, equality, hashing and the cheap value predicates.
//!
//! These are the operations a container calls on every insert and lookup, so a
//! fixed overhead here is multiplied by the size of the collection rather than
//! amortised over the width of the number.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::hash::{Hash, Hasher};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint, rug_uint_pairs};
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, arbi_uint, arbi_uint_pairs},
};

/// Total ordering of two equal-width values.
///
/// Equal widths are the hard case: the answer is decided by the most
/// significant differing limb rather than by the lengths, so the comparison
/// cannot short-circuit on size alone.
mod cmp {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).cmp(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).cmp(black_box(right)));
            }
        });
    }
}

/// Equality, which may answer from the lengths alone.
mod eq {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left) == black_box(right));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left) == black_box(right));
            }
        });
    }
}

/// Hashing, which must traverse the whole value.
mod hash {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            black_box(&value).hash(&mut hasher);
            let _output = black_box(hasher.finish());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            black_box(&value).hash(&mut hasher);
            let _output = black_box(hasher.finish());
        });
    }
}

/// The smaller of two values, materialised.
mod min {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).min(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).min(black_box(right)).clone());
            }
        });
    }
}

/// The larger of two values, materialised.
mod max {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).max(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).max(black_box(right)).clone());
            }
        });
    }
}

/// Clamp into an inclusive range.
mod clamp {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        let first_bound = arbi_uint(bits, 1);
        let second_bound = arbi_uint(bits, 9_999);
        let (lower, upper) = if first_bound <= second_bound {
            (first_bound, second_bound)
        } else {
            (second_bound, first_bound)
        };
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).clamp(black_box(&lower), black_box(&upper)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let first_bound = rug_uint(bits, 1);
        let second_bound = rug_uint(bits, 9_999);
        let (lower, upper) = if first_bound <= second_bound {
            (first_bound, second_bound)
        } else {
            (second_bound, first_bound)
        };
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(
                black_box(&value).clamp_ref(black_box(&lower), black_box(&upper)),
            ));
        });
    }
}

/// Parity, which reads one limb.
mod is_even {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _even = black_box(black_box(&value).is_even());
            let _odd = black_box(black_box(&value).is_odd());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _even = black_box(black_box(&value).is_even());
            let _odd = black_box(black_box(&value).is_odd());
        });
    }
}

/// Whether exactly one bit is set.
///
/// The counterpart is the population count a caller would compare against one.
mod is_power_of_two {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_power_of_two());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).count_ones() == Some(1));
        });
    }
}
