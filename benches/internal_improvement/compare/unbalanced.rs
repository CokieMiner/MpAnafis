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
use mp_anafis::tune_api::tier::{Limb, state::MultiplicationBenchState};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, FlintThreadBudget, assert_compatible_limb_width, flint_mpn_mul,
    },
    shared::{HUGE_SHAPES, SHAPES, gmp_pair_reference, operands_pair, validate_and_warm_product},
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
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "Mp unbalanced production product", |probe| {
        reusable.run(probe, &larger, &smaller);
    });
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = SHAPES)]
fn gmp_production_serial(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = size_t::try_from(larger_len).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller_len).expect("width fits a GMP size");
    let mut expected = vec![Limb::MIN; larger_len.saturating_add(smaller_len)];
    let mut oracle = MultiplicationBenchState::default();
    oracle.run(&mut expected, &larger, &smaller);
    validate_and_warm_product(&expected, "GMP unbalanced product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of their exact counts and the probe holds the complete product.
        unsafe {
            let _high_limb = gmp::mpn_mul(
                probe.as_mut_ptr().cast::<limb_t>(),
                larger.as_ptr().cast::<limb_t>(),
                larger_count,
                smaller.as_ptr().cast::<limb_t>(),
                smaller_count,
            );
        }
    });
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
fn flint_production_serial(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let _threads = FlintThreadBudget::new(1);
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = FlintSize::try_from(larger_len).expect("width fits a FLINT size");
    let smaller_count = FlintSize::try_from(smaller_len).expect("width fits a FLINT size");
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "FLINT unbalanced production product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of their exact counts and the probe holds the complete product.
        unsafe {
            let _high_limb = flint_mpn_mul(
                probe.as_mut_ptr().cast::<FlintLimb>(),
                larger.as_ptr().cast::<FlintLimb>(),
                larger_count,
                smaller.as_ptr().cast::<FlintLimb>(),
                smaller_count,
            );
        }
    });

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
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "Mp huge unbalanced product", |probe| {
        reusable.run(probe, &larger, &smaller);
    });
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn gmp_production_serial_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = size_t::try_from(larger_len).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller_len).expect("width fits a GMP size");
    let mut expected = vec![Limb::MIN; larger_len.saturating_add(smaller_len)];
    let mut oracle = MultiplicationBenchState::default();
    oracle.run(&mut expected, &larger, &smaller);
    validate_and_warm_product(&expected, "GMP huge unbalanced product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of their exact counts and the probe holds the complete product.
        unsafe {
            let _high_limb = gmp::mpn_mul(
                probe.as_mut_ptr().cast::<limb_t>(),
                larger.as_ptr().cast::<limb_t>(),
                larger_count,
                smaller.as_ptr().cast::<limb_t>(),
                smaller_count,
            );
        }
    });
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
fn flint_production_serial_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    const { assert_one_limb_width() }
    let _threads = FlintThreadBudget::new(1);
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = FlintSize::try_from(larger_len).expect("width fits a FLINT size");
    let smaller_count = FlintSize::try_from(smaller_len).expect("width fits a FLINT size");
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "FLINT huge unbalanced product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of their exact counts and the probe holds the complete product.
        unsafe {
            let _high_limb = flint_mpn_mul(
                probe.as_mut_ptr().cast::<FlintLimb>(),
                larger.as_ptr().cast::<FlintLimb>(),
                larger_count,
                smaller.as_ptr().cast::<FlintLimb>(),
                smaller_count,
            );
        }
    });
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
