//! Radix formatting and parsing.
//!
//! Several radices, because they select different algorithms rather than
//! different constants. Radix 10 needs a real division per digit group and a
//! divide-and-conquer split once the value is wide enough. Radices 2, 4, and
//! 16 map each source byte onto a fixed lookup row of whole digits, so the
//! conversion is a shift, mask, and table copy with no arithmetic at all.
//! Radices 8 and 32 extract digits from digit-aligned byte blocks: a three-
//! or five-byte block holds exactly eight digits, so each block yields
//! independent shift-and-mask digit extractions with no carry state.
//! Benchmarking only radix 10 hides whether these specialisations exist.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint},
};

mod to_string_radix_10 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(10));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(10));
        });
    }
}

mod from_string_radix_10 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let text = mp_uint(bits, 42).to_string_radix(10);
        bencher.bench_local(|| {
            let _output = black_box(MpUint::from_str_radix(black_box(&text), 10));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let text = rug_uint(bits, 42).to_string_radix(10);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_str_radix(black_box(&text), 10));
        });
    }
}

mod to_string_radix_16 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(16));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(16));
        });
    }
}

mod from_string_radix_16 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let text = mp_uint(bits, 42).to_string_radix(16);
        bencher.bench_local(|| {
            let _output = black_box(MpUint::from_str_radix(black_box(&text), 16));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let text = rug_uint(bits, 42).to_string_radix(16);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_str_radix(black_box(&text), 16));
        });
    }
}

mod to_string_radix_32 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(32));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(32));
        });
    }
}

mod to_string_radix_8 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(8));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(8));
        });
    }
}

mod to_string_radix_2 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(2));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(2));
        });
    }
}

mod from_string_radix_2 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let text = mp_uint(bits, 42).to_string_radix(2);
        bencher.bench_local(|| {
            let _output = black_box(MpUint::from_str_radix(black_box(&text), 2));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let text = rug_uint(bits, 42).to_string_radix(2);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_str_radix(black_box(&text), 2));
        });
    }
}

/// The `Display` implementation, which is decimal formatting through
/// `core::fmt` rather than through the radix entry point.
mod display {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string());
        });
    }
}

/// The `LowerHex` implementation.
mod lower_hex {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:x}", black_box(&value)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:x}", black_box(&value)));
        });
    }
}

/// The `Binary` implementation.
mod binary {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:b}", black_box(&value)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:b}", black_box(&value)));
        });
    }
}

/// The `Octal` implementation, the third power-of-two radix.
mod octal {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:o}", black_box(&value)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(format!("{:o}", black_box(&value)));
        });
    }
}
