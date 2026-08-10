//! Addressed reads and writes of one bit, plus the `bit_range` slice.
//!
//! Our writers return a new value; GMP's mutate in place. The `rug` halves
//! therefore clone first, so both sides pay for one allocation and the cell
//! compares the bit write rather than the calling convention.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]
#![allow(
    clippy::arithmetic_side_effects,
    reason = "the rug counterpart builds its range mask with the expression a caller would write by hand; rug exposes no checked forms to write it with"
)]

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::NARROW,
    support::{SAMPLE_SIZE_FAST, arbi_uint},
};

/// The bit these benchmarks address, inside the first limb on every supported
/// pointer width.
const TARGET_BIT: usize = 17;

mod get_bit {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).get_bit(TARGET_BIT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).get_bit(index));
        });
    }
}

/// `test_bit`, the same read under the name the inventory also lists.
mod test_bit {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).test_bit(TARGET_BIT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).get_bit(index));
        });
    }
}

mod set_bit {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).set_bit(TARGET_BIT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let mut updated = black_box(&value).clone();
            let _written = updated.set_bit(index, true);
            let _output = black_box(updated);
        });
    }
}

mod clear_bit {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).clear_bit(TARGET_BIT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let mut updated = black_box(&value).clone();
            let _written = updated.set_bit(index, false);
            let _output = black_box(updated);
        });
    }
}

mod toggle_bit {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).toggle_bit(TARGET_BIT));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let mut updated = black_box(&value).clone();
            let _written = updated.toggle_bit(index);
            let _output = black_box(updated);
        });
    }
}

mod set_bit_to {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).set_bit_to(TARGET_BIT, true));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let index = u32::try_from(TARGET_BIT).expect("the target bit fits in u32");
        bencher.bench_local(|| {
            let mut updated = black_box(&value).clone();
            let _written = updated.set_bit(index, true);
            let _output = black_box(updated);
        });
    }
}

/// Extract a contiguous span of bits.
///
/// The counterpart is the shift-and-mask a caller would write, with the mask
/// built once outside the timed region.
mod bit_range {
    use super::*;

    const RANGE_START: usize = 8;
    const RANGE_END: usize = 200;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).bit_range(RANGE_START, RANGE_END));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let start = u32::try_from(RANGE_START).expect("the range start fits in u32");
        let span = u32::try_from(RANGE_END.saturating_sub(RANGE_START))
            .expect("the range width fits in u32");
        let mask = Integer::from(Integer::u_pow_u(2, span)) - 1_u32;
        bencher.bench_local(|| {
            let shifted = Integer::from(black_box(&value) >> start);
            let _output = black_box(shifted & black_box(&mask));
        });
    }
}
