//! `mul_add`, `square`, `pow`, `midpoint`, `abs_diff`, `checked_next_power_of_two`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug midpoint counterpart is the add-then-shift a caller would write by hand; rug exposes no checked forms to write it with"
)]

use core::ops::{Add, Mul, Sub};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::Pow};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::{BALANCED, NARROW},
    support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_WIDE, arbi_uint, arbi_uint_pairs},
};

/// `a + b * c` as one operation.
///
/// GMP has no single-call fused form for arbitrary precision, so the
/// counterpart is the composed `mpz_add(a, mpz_mul(b, c))` a caller would
/// otherwise write. The gap is the temporary that composition materialises.
mod mul_add {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let addend = arbi_uint(bits, 42);
        let multiplier = arbi_uint(bits, 1_337);
        let multiplicand = arbi_uint(bits, 9_999);
        bencher.bench_local(|| {
            let _output = black_box(
                black_box(&addend).mul_add(black_box(&multiplier), black_box(&multiplicand)),
            );
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let addend = rug_uint(bits, 42);
        let multiplier = rug_uint(bits, 1_337);
        let multiplicand = rug_uint(bits, 9_999);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Add::add(
                black_box(&addend),
                Mul::mul(black_box(&multiplier), black_box(&multiplicand)),
            )));
        });
    }
}

/// `a * a` through the dedicated squaring path.
mod square {
    use super::*;

    #[divan::bench(args = BALANCED, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).square());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = BALANCED, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).square_ref()));
        });
    }
}

/// `a.pow(17)`: binary exponentiation over a small exponent.
mod pow {
    use super::*;

    const EXPONENT: u32 = 17;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let base = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&base).pow(EXPONENT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let base = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Pow::pow(black_box(&base), EXPONENT)));
        });
    }
}

/// `(a + b) / 2` without materialising the sum.
///
/// The GMP counterpart is that materialising form: add, then shift right.
mod midpoint {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let left = arbi_uint(bits, 42);
        let right = arbi_uint(bits, 1_337);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&left).midpoint(black_box(&right)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_uint(bits, 42);
        let right = rug_uint(bits, 1_337);
        bencher.bench_local(|| {
            let mut sum = Integer::from(Add::add(black_box(&left), black_box(&right)));
            sum >>= 1_u32;
            let _output = black_box(sum);
        });
    }
}

/// `|a - b|` without a sign detour.
mod abs_diff {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let left = arbi_uint(bits, 42);
        let right = arbi_uint(bits, 1_337);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&left).abs_diff(black_box(&right)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_uint(bits, 42);
        let right = rug_uint(bits, 1_337);
        bencher.bench_local(|| {
            let difference = Integer::from(Sub::sub(black_box(&left), black_box(&right)));
            let _output = black_box(difference.abs());
        });
    }
}

/// The smallest power of two at or above the value.
///
/// GMP has no counterpart; the composed equivalent would be a bit scan followed
/// by a shift, which is what the method already does internally, so pairing it
/// against that would compare the implementation with itself.
mod checked_next_power_of_two {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let inputs = arbi_uint_pairs(bits);
        bencher.bench_local(|| {
            for (value, _unused) in &inputs {
                let _output = black_box(black_box(value).checked_next_power_of_two());
            }
        });
    }
}
