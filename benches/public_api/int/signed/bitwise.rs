//! Signed bitwise operations.
//!
//! Negative operands are the interesting case: our representation is sign and
//! magnitude, GMP's is the same, and both must present two's complement
//! semantics for the bitwise operators. That conversion is the cost being
//! measured, so every operand here is negative.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{BitAnd, BitOr, BitXor, Shl, Shr};

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, rug_int_pairs};
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_int, mp_int_pairs},
};

/// Shift counts spanning sub-limb and cross-limb distances.
const SHIFTS: [usize; 2] = [31, 137];

mod bitand {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitAnd::bitand(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
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
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitOr::bitor(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
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
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(BitXor::bitxor(black_box(left), black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
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

mod shl {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let _output = black_box(Shl::shl(black_box(&value), black_box(shift)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let count = u32::try_from(shift).expect("small shift fits in u32");
                let _output =
                    black_box(Integer::from(Shl::shl(black_box(&value), black_box(count))));
            }
        });
    }
}

/// Right shift of a negative value, which rounds toward negative infinity in
/// both libraries and therefore needs a magnitude correction.
mod shr {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let _output = black_box(Shr::shr(black_box(&value), black_box(shift)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            for shift in SHIFTS {
                let count = u32::try_from(shift).expect("small shift fits in u32");
                let _output =
                    black_box(Integer::from(Shr::shr(black_box(&value), black_box(count))));
            }
        });
    }
}

/// Population count of a negative value, which is only defined against the
/// two's complement form.
mod count_ones {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).count_ones());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).count_ones());
        });
    }
}

mod trailing_zeros {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).trailing_zeros());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).find_one(0));
        });
    }
}

mod significant_bits {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).significant_bits());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).significant_bits());
        });
    }
}
