//! Rectangular production multiplication on explicit longer/shorter limb counts.
//! Each engine uses the same deterministic inputs and reusable destination.
//! Mp numeric scratch is prepared before timing; dispatch and external-library
//! allocations remain timed. Balanced controls share the same shape matrix.

#![expect(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::{Limb, MultiplicationBenchState};

use crate::shared::{
    HUGE_SHAPES, SHAPES, ShapeWorkerCase, ambient_workers, gmp_pair_reference, operands_pair,
    parallel_shape_cases, shape_cases, validate_and_warm_product, validated_gmp_counts,
};

use super::{FlintLimb, FlintSize, FlintThreadBudget, assert_one_limb_width, flint_mpn_mul};

#[divan::bench(args = shape_cases(SHAPES, ambient_workers()))]
fn mp(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_mp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = shape_cases(SHAPES, 1))]
fn gmp_production_serial(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_gmp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = shape_cases(SHAPES, 1))]
fn flint_production_serial(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_flint(bencher, case.larger_len, case.smaller_len, case.workers);
}

#[divan::bench(args = parallel_shape_cases(SHAPES))]
fn flint_production_parallel(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_flint(bencher, case.larger_len, case.smaller_len, case.workers);
}

// Huge rows use three samples of one product each.

#[divan::bench(
    args = shape_cases(HUGE_SHAPES, ambient_workers()),
    sample_count = 3,
    sample_size = 1,
)]
fn mp_huge(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_mp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = shape_cases(HUGE_SHAPES, 1), sample_count = 3, sample_size = 1)]
fn gmp_production_serial_huge(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_gmp(bencher, case.larger_len, case.smaller_len);
}

#[divan::bench(args = shape_cases(HUGE_SHAPES, 1), sample_count = 3, sample_size = 1)]
fn flint_production_serial_huge(bencher: divan::Bencher<'_, '_>, case: ShapeWorkerCase) {
    bench_flint(bencher, case.larger_len, case.smaller_len, case.workers);
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
    let mut expected = vec![Limb::MIN; destination.len()];
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
    assert!(
        larger_len >= smaller_len && smaller_len > 0,
        "FLINT requires larger >= smaller >= 1"
    );
    let larger_count =
        FlintSize::try_from(larger_len).expect("benchmark width must fit a FLINT size");
    let smaller_count =
        FlintSize::try_from(smaller_len).expect("benchmark width must fit a FLINT size");
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "FLINT unbalanced product", |probe| {
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
