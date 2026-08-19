//! Endian byte vector serialisation.
//!
//! Our methods return an owned `Vec<u8>`. GMP's `mpz_export` writes into a
//! caller-supplied buffer, so the `rug` halves allocate that buffer inside the
//! timed region; otherwise the comparison would charge one side for an
//! allocation the other never makes.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::Order};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint},
};

mod to_be_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_be_bytes());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let len = value.significant_digits::<u8>();
        bencher.bench_local(|| {
            let mut buffer = vec![0_u8; len];
            black_box(&value).write_digits(&mut buffer, Order::MsfBe);
            let _output = black_box(buffer);
        });
    }
}

mod from_be_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let bytes = mp_uint(bits, 42).to_be_bytes();
        bencher.bench_local(|| {
            let _output = black_box(MpUint::from_be_bytes(black_box(&bytes)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let mut bytes = vec![0_u8; value.significant_digits::<u8>()];
        value.write_digits(&mut bytes, Order::MsfBe);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_digits(black_box(&bytes), Order::MsfBe));
        });
    }
}

mod to_le_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).to_le_bytes());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let len = value.significant_digits::<u8>();
        bencher.bench_local(|| {
            let mut buffer = vec![0_u8; len];
            black_box(&value).write_digits(&mut buffer, Order::LsfLe);
            let _output = black_box(buffer);
        });
    }
}

mod from_le_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let bytes = mp_uint(bits, 42).to_le_bytes();
        bencher.bench_local(|| {
            let _output = black_box(MpUint::from_le_bytes(black_box(&bytes)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let mut bytes = vec![0_u8; value.significant_digits::<u8>()];
        value.write_digits(&mut bytes, Order::LsfLe);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from_digits(black_box(&bytes), Order::LsfLe));
        });
    }
}
