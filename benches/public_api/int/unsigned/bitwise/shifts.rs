//! `Shl`, `Shr` and the left-shift policy family.
//!
//! Each cell shifts by three counts — a sub-limb count, a count just under one
//! limb, and a count spanning two limbs — so neither the aligned fast path nor
//! the cross-limb merge can dominate on its own.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Shl, Shr};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint},
};

/// Sub-limb, near-limb and cross-limb shift counts.
const SHIFTS: [usize; 3] = [3, 31, 137];

mod shl {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let _output = black_box(Shl::shl(black_box(&value), black_box(shift)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let count = u32::try_from(shift).expect("small shift fits in u32");
                let _output =
                    black_box(Integer::from(Shl::shl(black_box(&value), black_box(count))));
            }
        });
    }
}

mod shr {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let _output = black_box(Shr::shr(black_box(&value), black_box(shift)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let count = u32::try_from(shift).expect("small shift fits in u32");
                let _output =
                    black_box(Integer::from(Shr::shr(black_box(&value), black_box(count))));
            }
        });
    }
}

/// The five shift policies together, against five plain GMP shifts.
///
/// They differ only in what they do when the result exceeds a bounded
/// precision, which unlimited operands never trigger, so one cell covering all
/// five reads more usefully than five near-identical ones.
mod shl_policies {
    use super::*;

    const SHIFT: usize = 137;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _checked = black_box(black_box(&value).checked_shl(SHIFT));
            let _wrapping = black_box(black_box(&value).wrapping_shl(SHIFT));
            let _overflowing = black_box(black_box(&value).overflowing_shl(SHIFT));
            let _saturating = black_box(black_box(&value).saturating_shl(SHIFT));
            let _try_shl = black_box(black_box(&value).try_shl(SHIFT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let count = u32::try_from(SHIFT).expect("small shift fits in u32");
        bencher.bench_local(|| {
            for _repetition in 0..5_u32 {
                let _output =
                    black_box(Integer::from(Shl::shl(black_box(&value), black_box(count))));
            }
        });
    }
}
