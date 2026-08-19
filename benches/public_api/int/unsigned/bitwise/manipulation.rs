//! Whole-value rearrangement: `reverse_bits`, `rotate_left`, `rotate_right`,
//! `swap_bytes`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug counterparts are the shift-and-mask expressions a caller would write by hand, on operands whose widths are fixed by the benchmark argument; rug exposes no checked forms to write them with"
)]

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, integer::Order};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, mp_uint},
};

/// Rotation distance, chosen to straddle a limb boundary.
const ROTATION: u32 = 17;

/// Reverse the bit order within an explicit width.
///
/// No `rug` half: GMP has no bit reversal, and the composed equivalent is a
/// per-bit loop calling `get_bit` and `set_bit`, which measures the loop rather
/// than any GMP kernel.
mod reverse_bits {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).reverse_bits(bits));
        });
    }
}

/// Rotate left within an explicit width.
///
/// The counterpart is the shift-or-shift-mask form a caller would write, which
/// is the only way to express a bounded rotation with GMP.
mod rotate_left {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).rotate_left(ROTATION, bits));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(Integer::u_pow_u(2, width)) - 1_u32;
        let counter_rotation = width.wrapping_sub(ROTATION);
        bencher.bench_local(|| {
            let high = Integer::from(black_box(&value) << ROTATION) & black_box(&mask);
            let low = Integer::from(black_box(&value) >> counter_rotation);
            let _output = black_box(high | low);
        });
    }
}

/// Rotate right within an explicit width.
mod rotate_right {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).rotate_right(ROTATION, bits));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let width = u32::try_from(bits).expect("benchmark widths fit in u32");
        let mask = Integer::from(Integer::u_pow_u(2, width)) - 1_u32;
        let counter_rotation = width.wrapping_sub(ROTATION);
        bencher.bench_local(|| {
            let low = Integer::from(black_box(&value) >> ROTATION);
            let high = Integer::from(black_box(&value) << counter_rotation) & black_box(&mask);
            let _output = black_box(high | low);
        });
    }
}

/// Reverse the byte order.
///
/// The counterpart writes the value out most-significant-byte-first and reads it
/// back least-significant-byte-first, which is what a byte swap is.
mod swap_bytes {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).swap_bytes());
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
            let _output = black_box(Integer::from_digits(&buffer, Order::LsfLe));
        });
    }
}
