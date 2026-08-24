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

#![expect(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::tier::{Limb, state::MultiplicationBenchState};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, FlintThreadBudget, assert_one_limb_width, flint_mpn_mul,
    },
    shared::{
        HUGE_SHAPES, SHAPES, ShapeWorkerCase, ambient_shape_cases, gmp_pair_reference,
        operands_pair, parallel_shape_cases, validate_and_warm_product, validated_gmp_counts,
    },
};

#[divan::bench(args = ambient_shape_cases(SHAPES))]
fn mp(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_mp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = SHAPES)]
fn gmp_production_serial(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    bench_gmp(bencher, larger_len, smaller_len);
}

#[divan::bench(args = SHAPES)]
fn flint_production_serial(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    bench_flint(bencher, larger_len, smaller_len, 1);
}

#[divan::bench(args = parallel_shape_cases(SHAPES))]
fn flint_production_parallel(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_flint(bencher, case.larger_len, case.smaller_len, case.workers);
}

// The huge arms below extend the matrix past the transform crossover, where
// `PLAN.md` currently reports every ratio but 1:16 as empty — a reading that
// rests on 32769 limbs being the widest shape ever measured. One product per
// sample, as in the huge production ladder.

#[divan::bench(
    args = ambient_shape_cases(HUGE_SHAPES),
    sample_count = 3,
    sample_size = 1,
)]
fn mp_huge(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_mp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn gmp_production_serial_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    bench_gmp(bencher, larger_len, smaller_len);
}

#[divan::bench(args = HUGE_SHAPES, sample_count = 3, sample_size = 1)]
fn flint_production_serial_huge(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    bench_flint(bencher, larger_len, smaller_len, 1);
}

#[divan::bench(
    args = parallel_shape_cases(HUGE_SHAPES),
    sample_count = 3,
    sample_size = 1,
)]
fn flint_production_parallel_huge(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_flint(bencher, case.larger_len, case.smaller_len, case.workers);
}

fn bench_mp(bencher: divan::Bencher<'_, '_>, larger_len: usize, smaller_len: usize) {
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "Mp unbalanced production product", |probe| {
        reusable.prepare(probe, &larger, &smaller).run();
    });
    let mut prepared = reusable.prepare(&mut destination, &larger, &smaller);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

fn bench_gmp(bencher: divan::Bencher<'_, '_>, larger_len: usize, smaller_len: usize) {
    const { assert_one_limb_width() }
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let (larger_count, smaller_count) = validated_gmp_counts(larger_len, smaller_len);
    let mut expected = vec![Limb::MIN; larger_len.saturating_add(smaller_len)];
    let mut oracle = MultiplicationBenchState::default();
    oracle.prepare(&mut expected, &larger, &smaller).run();
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
        // both operands hold their exact counts with the longer operand first,
        // and `destination` holds their complete product.
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

fn bench_flint(
    bencher: divan::Bencher<'_, '_>,
    larger_len: usize,
    smaller_len: usize,
    workers: usize,
) {
    const { assert_one_limb_width() }
    let _threads = FlintThreadBudget::new(workers);
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let (larger_count, smaller_count) = validated_flint_counts(larger_len, smaller_len);
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
        // SAFETY: the three vectors are independently allocated and disjoint,
        // both operands hold their exact counts with the longer operand first,
        // and `destination` holds their complete product.
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

fn validated_flint_counts(larger_len: usize, smaller_len: usize) -> (FlintSize, FlintSize) {
    let larger = FlintSize::try_from(larger_len);
    let smaller = FlintSize::try_from(smaller_len);
    assert!(
        larger.is_ok() && smaller.is_ok(),
        "benchmark widths must fit FLINT sizes"
    );
    // SAFETY: the assertion above validates both conversions before timing.
    unsafe { (larger.unwrap_unchecked(), smaller.unwrap_unchecked()) }
}
