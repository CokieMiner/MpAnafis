//! `BitAnd`, `BitOr`, `BitXor` and the width-bounded complements
//! `not_with_width` and `try_not`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug counterpart builds its width mask with the expression a caller would write by hand; rug exposes no checked forms to write it with"
)]

use core::ops::{BitAnd, BitOr, BitXor};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_uint, rug_uint_pairs};
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, bounded_mp_uint, mp_uint, mp_uint_pairs},
};

mod bitand {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitAnd::bitand(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(BitAnd::bitand(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod bitor {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitOr::bitor(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(BitOr::bitor(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

mod bitxor {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitXor::bitxor(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(BitXor::bitxor(
                    black_box(left),
                    black_box(right),
                )));
            }
        });
    }
}

/// The complement within an explicit width.
///
/// GMP's `mpz_com` complements against an infinite two's complement sign
/// extension and returns a negative number, which is a different function. The
/// counterpart is therefore the masked form a caller would write to get the same
/// answer: `(2^width - 1) - value`. The mask is built once, outside the timed
/// region, exactly as our implementation gets its width for free.
mod not_with_width {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).not_with_width(bits));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(Integer::u_pow_u(2, width)) - 1_u32;
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&mask) - black_box(&value)));
        });
    }
}

/// The same complement, taking its width from the value's own bounded
/// precision instead of an argument.
///
/// The operand is bounded rather than unlimited, so the width lookup succeeds
/// and the cell measures the complement; on an unlimited operand this would
/// return `WidthRequired` without doing any work.
mod try_not {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = bounded_mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).try_not());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(Integer::u_pow_u(2, width)) - 1_u32;
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&mask) - black_box(&value)));
        });
    }
}
