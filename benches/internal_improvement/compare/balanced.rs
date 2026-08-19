//! Equal-width production-dispatch products against GMP and FLINT.
//!
//! Transform executor, geometry, and allocation policies live in the separate
//! `transform_matrix` group so no forced tier is mistaken for production here.

#![allow(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::{fmt, hint::black_box};

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    state::MultiplicationBenchState,
    transform::{DefaultExecutor, ParallelExecutor},
};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, FlintThreadBudget, assert_compatible_limb_width, flint_mpn_mul_n,
    },
    shared::{HUGE_SIZES, SCALING_SIZES, gmp_equal_reference, operands, validate_and_warm_product},
};

#[derive(Clone, Copy, Debug)]
struct FlintParallelCase {
    len: usize,
    workers: usize,
}

impl fmt::Display for FlintParallelCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}-limbs/{}-workers", self.len, self.workers)
    }
}

/// Asserts the limb widths all three arms are indexed by agree.
const fn assert_one_limb_width() {
    assert!(
        size_of::<Limb>() == size_of::<limb_t>(),
        "the GMP comparison requires one limb width"
    );
    assert_compatible_limb_width();
}

#[divan::bench(args = SCALING_SIZES)]
fn gmp_production_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands(len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    let mut expected = vec![Limb::MIN; left.len().saturating_mul(2)];
    let mut oracle = MultiplicationBenchState::default();
    oracle.run(&mut expected, &left, &right);
    validate_and_warm_product(&expected, "GMP balanced product", |probe| {
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
fn flint_production_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    benchmark_flint_production(bencher, len, 1);
}

#[divan::bench(args = flint_parallel_cases(SCALING_SIZES))]
fn flint_production_default_budget(bencher: divan::Bencher<'_, '_>, case: FlintParallelCase) {
    benchmark_flint_production(bencher, case.len, case.workers);
}

#[divan::bench(args = SCALING_SIZES)]
fn mp_production_reusable_scratch(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "Mp production product", |probe| {
        reusable.run(probe, &left, &right);
    });
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
    });
}

#[divan::bench(args = HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn gmp_production_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands(len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    let mut expected = vec![Limb::MIN; left.len().saturating_mul(2)];
    let mut oracle = MultiplicationBenchState::default();
    oracle.run(&mut expected, &left, &right);
    validate_and_warm_product(&expected, "GMP huge balanced product", |probe| {
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
fn flint_production_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    benchmark_flint_production(bencher, len, 1);
}

#[divan::bench(
    args = flint_parallel_cases(HUGE_SIZES),
    sample_count = 3,
    sample_size = 1,
)]
fn flint_production_default_budget_huge(bencher: divan::Bencher<'_, '_>, case: FlintParallelCase) {
    benchmark_flint_production(bencher, case.len, case.workers);
}

#[divan::bench(args = HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn mp_production_reusable_scratch_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut reusable = MultiplicationBenchState::default();
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "Mp huge production product", |probe| {
        reusable.run(probe, &left, &right);
    });
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
    });
}

fn benchmark_flint_production(bencher: divan::Bencher<'_, '_>, len: usize, workers: usize) {
    const { assert_one_limb_width() }
    let threads = FlintThreadBudget::new(workers);
    debug_assert_eq!(
        threads.workers(),
        workers,
        "FLINT guard must retain the benchmark's labeled worker budget"
    );
    let (left, right, mut destination) = operands(len);
    let count = FlintSize::try_from(len).expect("width fits a FLINT size");
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "FLINT balanced production product", |probe| {
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

fn flint_parallel_cases<const N: usize>(sizes: [usize; N]) -> Vec<FlintParallelCase> {
    let workers = DefaultExecutor::default().parallelism().get();
    if workers <= 1 {
        return Vec::new();
    }
    sizes
        .into_iter()
        .map(|len| FlintParallelCase { len, workers })
        .collect()
}
