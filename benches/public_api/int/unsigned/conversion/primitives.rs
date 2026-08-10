//! Casts to and from the fixed-width primitives.
//!
//! Two argument widths throughout: one that fits the target type, so the cast
//! succeeds and does the work, and one that does not, so it takes the range
//! check and returns `None`. A single in-range cell would leave the rejection
//! path unmeasured.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use arbi_anafis::ArbiUint;
use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::support::{SAMPLE_SIZE_FAST, arbi_uint};

/// Widths straddling the 64-bit boundary.
const WORD_BITS: [usize; 2] = [64, 256];

/// Widths straddling the 128-bit boundary.
const DOUBLE_WORD_BITS: [usize; 2] = [128, 256];

mod to_u64 {
    use super::*;

    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_u64());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_u64());
        });
    }
}

mod to_i64 {
    use super::*;

    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i64());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i64());
        });
    }
}

mod to_u128 {
    use super::*;

    #[divan::bench(args = DOUBLE_WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_u128());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DOUBLE_WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_u128());
        });
    }
}

mod to_i128 {
    use super::*;

    #[divan::bench(args = DOUBLE_WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i128());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = DOUBLE_WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i128());
        });
    }
}

/// `to_usize` and `to_isize`, the pointer-width pair.
mod to_size {
    use super::*;

    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _unsigned = black_box(black_box(&value).to_usize());
            let _signed = black_box(black_box(&value).to_isize());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = WORD_BITS, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _unsigned = black_box(black_box(&value).to_usize());
            let _signed = black_box(black_box(&value).to_isize());
        });
    }
}

/// Rounding to `f64`, which must find the top 53 bits and the sticky remainder.
mod to_f64 {
    use super::*;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f64());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f64());
        });
    }
}

mod to_f32 {
    use super::*;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f32());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f32());
        });
    }
}

/// Construction from a primitive, the direction the `From` implementations
/// cover.
mod from_u64 {
    use super::*;

    const SEED_VALUE: u64 = 0xdead_beef_cafe_f00d;

    #[divan::bench(sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher) {
        bencher.bench_local(|| {
            let _output = black_box(ArbiUint::from(black_box(SEED_VALUE)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher) {
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(SEED_VALUE)));
        });
    }
}
