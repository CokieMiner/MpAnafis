//! The operand-shape matrix: Mp against GMP and FLINT across seven ratios.
//!
//! Reported per ratio and per size band so a missing algorithm can be told from
//! a mistuned crossover: a ratio behind in *every* band is a shape nothing
//! serves, while a ratio behind only near a crossover is a threshold to move.
//!
//! Both operands are exact-width and the destination is caller-owned, and Mp
//! runs through the full public tower rather than any forced tier, so what is
//! measured is the dispatcher's own shape decision.
//!
//! Mp's scratch pool is acquired once per shape rather than per iteration.
//! GMP allocates its own workspace on the stack, so timing our pool acquisition
//! inside the loop compares an allocator against an algorithm — and its variance
//! is not small: it made the 200- and 400-limb columns move by up to 47% between
//! runs of one binary.

#![allow(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{Limb, state::MulBenchScratch};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, assert_compatible_limb_width, flint_mpn_mul, pin_to_one_thread,
    },
    shared::{HUGE_SHAPES, SHAPES, operands_pair},
};

/// Asserts the limb widths all three arms are indexed by agree.
const fn assert_one_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<limb_t>(),
        "the GMP comparison requires one limb width"
    );
    assert_compatible_limb_width();
}

#[divan::bench(args = SHAPES)]
fn mp(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut reusable = MulBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = SHAPES)]
fn gmp(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = size_t::try_from(larger_len).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller_len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: the three vectors are independently allocated and disjoint,
        // `larger` and `smaller` hold exactly their stated limb counts with the
        // longer operand first as mpn_mul requires, and `destination` holds
        // their sum.
        let _high = unsafe {
            gmp::mpn_mul(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(larger.as_ptr().cast::<limb_t>()),
                black_box(larger_count),
                black_box(smaller.as_ptr().cast::<limb_t>()),
                black_box(smaller_count),
            )
        };
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = SHAPES)]
fn flint(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    pin_to_one_thread();
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = FlintSize::try_from(larger_len).expect("width fits a FLINT size");
    let smaller_count = FlintSize::try_from(smaller_len).expect("width fits a FLINT size");

    let mut expected = vec![Limb::MIN; destination.len()];
    // SAFETY: as in the timed block below, with `expected` in place of
    // `destination`; both hold `larger_len + smaller_len` limbs.
    unsafe {
        let _gmp_high = gmp::mpn_mul(
            expected.as_mut_ptr().cast::<limb_t>(),
            larger.as_ptr().cast::<limb_t>(),
            size_t::try_from(larger_len).expect("width fits a GMP size"),
            smaller.as_ptr().cast::<limb_t>(),
            size_t::try_from(smaller_len).expect("width fits a GMP size"),
        );
        let _flint_high = flint_mpn_mul(
            destination.as_mut_ptr().cast::<FlintLimb>(),
            larger.as_ptr().cast::<FlintLimb>(),
            larger_count,
            smaller.as_ptr().cast::<FlintLimb>(),
            smaller_count,
        );
    }
    assert_eq!(
        destination, expected,
        "FLINT multiplication disagrees with GMP"
    );

    bencher.bench_local(|| {
        // SAFETY: the three vectors are independently allocated and disjoint,
        // both operands hold exactly their stated limb counts with the longer
        // first as `flint_mpn_mul` requires, and `destination` holds their sum.
        let _high = unsafe {
            flint_mpn_mul(
                black_box(destination.as_mut_ptr().cast::<FlintLimb>()),
                black_box(larger.as_ptr().cast::<FlintLimb>()),
                black_box(larger_count),
                black_box(smaller.as_ptr().cast::<FlintLimb>()),
                black_box(smaller_count),
            )
        };
        let _output = black_box(&destination);
    });
}

// The huge arms below extend the matrix past the transform crossover, where
// `PLAN.md` currently reports every ratio but 1:16 as empty — a reading that
// rests on 32769 limbs being the widest shape ever measured. One product per
// sample, as in `balanced`.

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn mp_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut reusable = MulBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn gmp_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = size_t::try_from(larger_len).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller_len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: as in `gmp` above.
        let _high = unsafe {
            gmp::mpn_mul(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(larger.as_ptr().cast::<limb_t>()),
                black_box(larger_count),
                black_box(smaller.as_ptr().cast::<limb_t>()),
                black_box(smaller_count),
            )
        };
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn flint_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    pin_to_one_thread();
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = FlintSize::try_from(larger_len).expect("width fits a FLINT size");
    let smaller_count = FlintSize::try_from(smaller_len).expect("width fits a FLINT size");
    bencher.bench_local(|| {
        // SAFETY: as in `flint` above.
        let _high = unsafe {
            flint_mpn_mul(
                black_box(destination.as_mut_ptr().cast::<FlintLimb>()),
                black_box(larger.as_ptr().cast::<FlintLimb>()),
                black_box(larger_count),
                black_box(smaller.as_ptr().cast::<FlintLimb>()),
                black_box(smaller_count),
            )
        };
        let _output = black_box(&destination);
    });
}
