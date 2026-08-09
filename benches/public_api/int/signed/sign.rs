//! Sign manipulation and inspection: the part of `MpInt` with no unsigned
//! counterpart.
//!
//! Every operand is negative, because a positive one takes the branch that does
//! nothing.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug counterparts are the subtract-then-correct expressions a caller would write by hand; rug exposes no checked forms to write them with"
)]

use core::ops::Neg;

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_int},
};

/// `|a|`, allocating a fresh value.
mod abs {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).abs());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).abs_ref()));
        });
    }
}

/// `|a|` in place, which should not touch the limbs at all.
mod abs_assign {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let mut target = black_box(&value).clone();
            target.abs_assign();
            let _output = black_box(target);
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let mut target = black_box(&value).clone();
            target.abs_mut();
            let _output = black_box(target);
        });
    }
}

/// The checked absolute value, which cannot fail without a fixed width but
/// still pays for asking.
mod checked_abs {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).checked_abs());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).abs_ref()));
        });
    }
}

/// `-a`.
mod neg {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Neg::neg(black_box(&value)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Neg::neg(black_box(&value))));
        });
    }
}

/// The sign as a value.
mod signum {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).signum());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).signum_ref()));
        });
    }
}

/// The magnitude as an `MpUint`, dropping the sign rather than clearing it.
mod unsigned_abs {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).unsigned_abs());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).abs_ref()));
        });
    }
}

/// The positive difference `max(0, a - b)`, which is `num_traits::Signed`'s
/// `abs_sub` and *not* the absolute difference — see [`abs_diff`] for that.
///
/// The left operand is the larger one, so the subtraction actually runs; with
/// the operands the other way round the method returns zero from the leading
/// comparison and the cell would measure that comparison.
mod abs_sub {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_int(bits, 42, false);
        let right = mp_int(bits, 1_337, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&left).abs_sub(black_box(&right)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_int(bits, 42, false);
        let right = rug_int(bits, 1_337, true);
        bencher.bench_local(|| {
            let difference = Integer::from(black_box(&left) - black_box(&right));
            let _output = black_box(difference.max(Integer::ZERO));
        });
    }
}

/// `abs_diff`, which returns an unsigned magnitude directly.
mod abs_diff {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_int(bits, 42, true);
        let right = mp_int(bits, 1_337, false);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&left).abs_diff(black_box(&right)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_int(bits, 42, true);
        let right = rug_int(bits, 1_337, false);
        bencher.bench_local(|| {
            let difference = Integer::from(black_box(&left) - black_box(&right));
            let _output = black_box(difference.abs());
        });
    }
}

/// The three sign predicates, which read a field.
mod predicates {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _negative = black_box(black_box(&value).is_negative());
            let _positive = black_box(black_box(&value).is_positive());
            let _minus_one = black_box(black_box(&value).is_minus_one());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _negative = black_box(black_box(&value) < &Integer::ZERO);
            let _positive = black_box(black_box(&value) > &Integer::ZERO);
            let _minus_one = black_box(black_box(&value) == &-1_i32);
        });
    }
}
