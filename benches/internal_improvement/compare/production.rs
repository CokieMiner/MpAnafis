//! Equal-width production multiplication with reusable output and numeric scratch.
//! Each engine checks one untimed product against an independent reference.
//! Standard and huge rows retain separate sampling and labeled worker budgets.

#![expect(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::{Limb, MultiplicationBenchState};

use crate::shared::{
    PRODUCTION_COMPARE_HUGE_SIZES, PRODUCTION_COMPARE_SIZES, WorkerCase, ambient_workers,
    gmp_equal_reference, operands_pair, parallel_worker_cases, validate_and_warm_product,
    validated_gmp_count, worker_cases,
};

use super::{FlintLimb, FlintSize, FlintThreadBudget, assert_one_limb_width, flint_mpn_mul_n};

#[divan::bench(args = worker_cases(PRODUCTION_COMPARE_SIZES, ambient_workers()))]
fn mp(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_mp(bencher, case.len);
}

#[divan::bench(
    args = worker_cases(PRODUCTION_COMPARE_HUGE_SIZES, ambient_workers()),
    sample_count = 3,
    sample_size = 1,
)]
fn mp_huge(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_mp(bencher, case.len);
}

#[divan::bench(args = worker_cases(PRODUCTION_COMPARE_SIZES, 1))]
fn gmp_serial(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_gmp(bencher, case.len);
}

#[divan::bench(args = worker_cases(PRODUCTION_COMPARE_HUGE_SIZES, 1), sample_count = 3, sample_size = 1)]
fn gmp_serial_huge(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_gmp(bencher, case.len);
}

#[divan::bench(args = worker_cases(PRODUCTION_COMPARE_SIZES, 1))]
fn flint_serial(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_flint(bencher, case.len, case.workers);
}

#[divan::bench(args = worker_cases(PRODUCTION_COMPARE_HUGE_SIZES, 1), sample_count = 3, sample_size = 1)]
fn flint_serial_huge(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_flint(bencher, case.len, case.workers);
}

#[divan::bench(args = parallel_worker_cases(PRODUCTION_COMPARE_SIZES))]
fn flint_parallel(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_flint(bencher, case.len, case.workers);
}

#[divan::bench(
    args = parallel_worker_cases(PRODUCTION_COMPARE_HUGE_SIZES),
    sample_count = 3,
    sample_size = 1,
)]
fn flint_parallel_huge(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_flint(bencher, case.len, case.workers);
}

fn bench_mp(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (left, right, mut destination) = operands_pair(len, len);
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "Mp production product", |probe| {
        reusable.prepare(probe, &left, &right).run();
    });
    let mut prepared = reusable.prepare(&mut destination, &left, &right);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}

fn bench_gmp(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands_pair(len, len);
    let count = validated_gmp_count(len);
    let mut expected = vec![Limb::MIN; destination.len()];
    let mut oracle = MultiplicationBenchState::default();
    oracle.prepare(&mut expected, &left, &right).run();
    validate_and_warm_product(&expected, "GMP production product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of exactly `count` limbs and the probe holds the complete product.
        unsafe {
            gmp::mpn_mul_n(
                probe.as_mut_ptr().cast::<limb_t>(),
                left.as_ptr().cast::<limb_t>(),
                right.as_ptr().cast::<limb_t>(),
                count,
            );
        }
    });
    bencher.bench_local(|| {
        // SAFETY: the three vectors are disjoint initialized spans, both inputs
        // hold `count` limbs, and the destination holds their complete product.
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

fn bench_flint(bencher: divan::Bencher<'_, '_>, len: usize, workers: usize) {
    const { assert_one_limb_width() }
    let _threads = FlintThreadBudget::new(workers);
    let (left, right, mut destination) = operands_pair(len, len);
    assert!(len > 0, "FLINT requires nonempty operands");
    let count = FlintSize::try_from(len).expect("benchmark width must fit a FLINT size");
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "FLINT production product", |probe| {
        // SAFETY: the probe and both inputs are independent, initialized spans
        // of exactly `count` limbs and the probe holds the complete product.
        unsafe {
            flint_mpn_mul_n(
                probe.as_mut_ptr().cast::<FlintLimb>(),
                left.as_ptr().cast::<FlintLimb>(),
                right.as_ptr().cast::<FlintLimb>(),
                count,
            );
        }
    });
    bencher.bench_local(|| {
        // SAFETY: the three vectors are disjoint initialized spans, both inputs
        // hold `count` limbs, and the destination holds their complete product.
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
