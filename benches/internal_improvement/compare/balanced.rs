//! Equal-width products: Arbi against GMP and FLINT on a log-uniform ladder.
//!
//! Every arm takes identical operands from the shared generator and writes into
//! a caller-owned destination, and the Arbi arm runs the full public dispatcher
//! rather than a forced tier, so what is compared is three complete towers.
//!
//! Arbi's scratch pool is acquired once per width, outside the timed region.
//! GMP and FLINT allocate their own workspace internally, so timing our pool
//! acquisition per iteration would put an allocator on one side of the
//! comparison and an algorithm on the other.

#![allow(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{Limb, state::MulBenchScratch};
use gmp_mpfr_sys::gmp::{self, limb_t, size_t};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, assert_compatible_limb_width, flint_mpn_mul_n, pin_to_one_thread,
    },
    shared::{HUGE_SIZES, SCALING_SIZES, operands},
};

/// Asserts the limb widths all three arms are indexed by agree.
const fn assert_one_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<limb_t>(),
        "the GMP comparison requires one limb width"
    );
    assert_compatible_limb_width();
}

#[divan::bench(args = SCALING_SIZES)]
fn arbi(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut reusable = MulBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
    });
}

#[divan::bench(args = SCALING_SIZES)]
fn gmp(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands(len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: the three vectors are independently allocated and therefore
        // disjoint, both operands hold exactly `count` initialized limbs, and
        // `destination` holds their sum.
        unsafe {
            gmp::mpn_mul_n(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(left.as_ptr().cast::<limb_t>()),
                black_box(right.as_ptr().cast::<limb_t>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = SCALING_SIZES)]
fn flint(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    pin_to_one_thread();
    let (left, right, mut destination) = operands(len);
    let count = FlintSize::try_from(len).expect("width fits a FLINT size");

    let mut expected = vec![Limb::MIN; destination.len()];
    // SAFETY: as in the timed block below, with `expected` in place of
    // `destination` and both holding `2 * len` limbs.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr().cast::<limb_t>(),
            left.as_ptr().cast::<limb_t>(),
            right.as_ptr().cast::<limb_t>(),
            size_t::try_from(len).expect("width fits a GMP size"),
        );
        flint_mpn_mul_n(
            destination.as_mut_ptr().cast::<FlintLimb>(),
            left.as_ptr().cast::<FlintLimb>(),
            right.as_ptr().cast::<FlintLimb>(),
            count,
        );
    }
    assert_eq!(
        destination, expected,
        "FLINT multiplication disagrees with GMP"
    );

    bencher.bench_local(|| {
        // SAFETY: the three vectors are independently allocated and therefore
        // disjoint, both operands hold exactly `count` initialized limbs, and
        // `destination` holds their sum.
        unsafe {
            flint_mpn_mul_n(
                black_box(destination.as_mut_ptr().cast::<FlintLimb>()),
                black_box(left.as_ptr().cast::<FlintLimb>()),
                black_box(right.as_ptr().cast::<FlintLimb>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}

// The huge arms below pin one product per sample and take few samples: at the
// top width a single product runs long enough that divan's default sampling
// would keep the machine busy for hours. They also skip the cross-library
// correctness check the ladder arms carry, because a verifying GMP product at
// 16.7 million limbs costs more than the measurement it guards, and the ladder
// arms already establish agreement on the same code paths.

#[divan::bench(args = HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn arbi_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut reusable = MulBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
    });
}

#[divan::bench(args = HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn gmp_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands(len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: as in `gmp` above.
        unsafe {
            gmp::mpn_mul_n(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(left.as_ptr().cast::<limb_t>()),
                black_box(right.as_ptr().cast::<limb_t>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn flint_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    pin_to_one_thread();
    let (left, right, mut destination) = operands(len);
    let count = FlintSize::try_from(len).expect("width fits a FLINT size");
    bencher.bench_local(|| {
        // SAFETY: as in `flint` above.
        unsafe {
            flint_mpn_mul_n(
                black_box(destination.as_mut_ptr().cast::<FlintLimb>()),
                black_box(left.as_ptr().cast::<FlintLimb>()),
                black_box(right.as_ptr().cast::<FlintLimb>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}
