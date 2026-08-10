//! Signed conversion and serialisation.
//!
//! Negative operands throughout: the sign is an extra character in every string
//! form and, in the byte forms, decides the whole encoding.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug counterpart negates a decoded magnitude, which is the sign restoration a caller must write by hand because GMP exports magnitudes only"
)]

use arbi_anafis::ArbiInt;
use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_int;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, arbi_int},
};

mod to_string_radix_10 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(10));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(10));
        });
    }
}

mod from_string_radix_10 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let text = arbi_int(bits, 42, true).to_string_radix(10);
        bencher.bench_local(|| {
            let _output = black_box(ArbiInt::from_str_radix(black_box(&text), 10));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let text = rug_int(bits, 42, true).to_string_radix(10);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_str_radix(black_box(&text), 10));
        });
    }
}

mod to_string_radix_16 {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(16));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_string_radix(16));
        });
    }
}

mod to_be_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_be_bytes());
        });
    }

    /// GMP exports magnitudes only, so the counterpart writes the magnitude and
    /// carries the sign separately, which is what a caller must do.
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        use rug::integer::Order;

        let value = rug_int(bits, 42, true);
        let len = value.significant_digits::<u8>();
        bencher.bench_local(|| {
            let mut buffer = vec![0_u8; len];
            black_box(&value).write_digits(&mut buffer, Order::MsfBe);
            let _output = black_box((buffer, black_box(&value) < &Integer::ZERO));
        });
    }
}

mod from_be_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let bytes = arbi_int(bits, 42, true).to_be_bytes();
        bencher.bench_local(|| {
            let _output = black_box(ArbiInt::from_be_bytes(black_box(&bytes)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        use rug::integer::Order;

        let value = rug_int(bits, 42, true);
        let mut bytes = vec![0_u8; value.significant_digits::<u8>()];
        value.write_digits(&mut bytes, Order::MsfBe);
        bencher.bench_local(|| {
            let magnitude = Integer::from_digits(black_box(&bytes), Order::MsfBe);
            let _output = black_box(-magnitude);
        });
    }
}

mod to_i64 {
    use super::*;

    #[divan::bench(args = [64, 256], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i64());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [64, 256], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_i64());
        });
    }
}

mod to_f64 {
    use super::*;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f64());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, true);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_f64());
        });
    }
}
