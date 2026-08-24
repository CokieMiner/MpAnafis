//! Full production multiplication towers against GMP and FLINT.
//!
//! Every Mp row runs the configured dispatcher with reusable destination and
//! scratch state; no row forces an algorithm tier. The ordinary ladder spans
//! basecase through cache-sized SSA products and samples exact dispatch
//! neighbors. The huge ladder continues to 2 Gibit per operand. Run the binary
//! inside a one-worker or wider ambient Rayon pool to control Mp; FLINT rows
//! report the same ambient worker budget as part of every parallel case label.

#![expect(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::tier::{Limb, state::MultiplicationBenchState};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, FlintThreadBudget, assert_one_limb_width, flint_mpn_mul_n,
    },
    shared::{
        PRODUCTION_COMPARE_HUGE_SIZES, PRODUCTION_COMPARE_SIZES, WorkerCase, ambient_worker_cases,
        gmp_equal_reference, operands, parallel_worker_cases, validate_and_warm_product,
        validated_gmp_count,
    },
};

#[divan::bench(args = ambient_worker_cases(PRODUCTION_COMPARE_SIZES))]
fn mp(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_mp(bencher, case.len);
}

#[divan::bench(
    args = ambient_worker_cases(PRODUCTION_COMPARE_HUGE_SIZES),
    sample_count = 3,
    sample_size = 1,
)]
fn mp_huge(bencher: divan::Bencher<'_, '_>, case: WorkerCase) {
    bench_mp(bencher, case.len);
}

#[divan::bench(args = PRODUCTION_COMPARE_SIZES)]
fn gmp_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_gmp(bencher, len);
}

#[divan::bench(args = PRODUCTION_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn gmp_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_gmp(bencher, len);
}

#[divan::bench(args = PRODUCTION_COMPARE_SIZES)]
fn flint_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_flint(bencher, len, 1);
}

#[divan::bench(args = PRODUCTION_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn flint_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_flint(bencher, len, 1);
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
    let (left, right, mut destination) = operands(len);
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
    let (left, right, mut destination) = operands(len);
    let count = validated_gmp_count(len);
    let mut expected = vec![Limb::MIN; left.len().saturating_mul(2)];
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
    let threads = FlintThreadBudget::new(workers);
    debug_assert_eq!(
        threads.workers(),
        workers,
        "FLINT guard must retain the benchmark's labeled worker budget"
    );
    let (left, right, mut destination) = operands(len);
    let count = validated_flint_count(len);
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

fn validated_flint_count(len: usize) -> FlintSize {
    let count = FlintSize::try_from(len);
    assert!(count.is_ok(), "benchmark width must fit a FLINT size");
    // SAFETY: the assertion above validates the conversion before timing.
    unsafe { count.unwrap_unchecked() }
}
