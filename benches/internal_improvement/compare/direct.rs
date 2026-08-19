//! Direct algorithm-vs-adversary benchmarks on dense size ladders for continuous graphing.
//!
//! Compares raw transform engines without tower dispatch:
//! - `MpAnafis` NTT (Sequential & Parallel)
//! - `MpAnafis` SSA (Sequential & Parallel)
//! - GMP `mpn_mul_n` (Serial baseline)
//! - FLINT `flint_mpn_mul_n` (Parallel baseline)

#![allow(
    unsafe_code,
    reason = "the benchmark calls raw GMP and FLINT mpn routines with disjoint, exactly sized vectors"
)]

use core::{fmt, hint::black_box};

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb, Tuner,
    transform::{
        DefaultExecutor, NttPlanPolicy, NttScratchPolicy, ParallelExecutor, SsaGeometryPolicy,
        SsaScratchPolicy, TransformExecutor,
    },
};

use crate::{
    compare::flint::{
        FlintLimb, FlintSize, FlintThreadBudget, assert_compatible_limb_width, flint_mpn_mul_n,
    },
    shared::{
        DENSE_COMPARE_HUGE_SIZES, DENSE_COMPARE_NTT_HUGE_SIZES, DENSE_COMPARE_SIZES,
        gmp_equal_reference, operands, validate_and_warm_product,
    },
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

// ── 1. MpAnafis NTT Sequential & Parallel ───────────────────────────────────

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn ntt_sequential(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ntt(bencher, len, TransformExecutor::Sequential);
}

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn ntt_parallel(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ntt(bencher, len, TransformExecutor::Default);
}

#[divan::bench(args = DENSE_COMPARE_NTT_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn ntt_sequential_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ntt(bencher, len, TransformExecutor::Sequential);
}

#[divan::bench(args = DENSE_COMPARE_NTT_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn ntt_parallel_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ntt(bencher, len, TransformExecutor::Default);
}

fn bench_ntt(bencher: divan::Bencher<'_, '_>, len: usize, executor: TransformExecutor) {
    let (left, right, mut destination) = operands(len);
    let expected = gmp_equal_reference(&left, &right);
    let Some(mut runner) = Tuner::bench_ntt_multiplication(
        NttPlanPolicy::Production,
        executor,
        NttScratchPolicy::Reusable,
        &left,
        &right,
    ) else {
        return;
    };
    validate_and_warm_product(&expected, "prepared NTT tier product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

// ── 2. MpAnafis SSA Sequential & Parallel ───────────────────────────────────

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn ssa_sequential(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa(bencher, len, TransformExecutor::Sequential);
}

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn ssa_parallel(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa(bencher, len, TransformExecutor::Default);
}

#[divan::bench(args = DENSE_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn ssa_sequential_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa(bencher, len, TransformExecutor::Sequential);
}

#[divan::bench(args = DENSE_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn ssa_parallel_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    bench_ssa(bencher, len, TransformExecutor::Default);
}

fn bench_ssa(bencher: divan::Bencher<'_, '_>, len: usize, executor: TransformExecutor) {
    let (left, right, mut destination) = operands(len);
    let expected = gmp_equal_reference(&left, &right);
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        SsaGeometryPolicy::Production,
        executor,
        SsaScratchPolicy::Reusable,
        &left,
        &right,
    ) else {
        return;
    };
    validate_and_warm_product(&expected, "prepared SSA tier product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

// ── 3. GMP Serial Reference ──────────────────────────────────────────────────

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn gmp_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    const { assert_one_limb_width() }
    let (left, right, mut destination) = operands(len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    let mut expected = vec![Limb::MIN; left.len().saturating_mul(2)];
    // SAFETY: the probe and input slices are valid non-aliasing spans of len limbs.
    unsafe {
        gmp::mpn_mul_n(
            expected.as_mut_ptr().cast::<limb_t>(),
            left.as_ptr().cast::<limb_t>(),
            right.as_ptr().cast::<limb_t>(),
            count,
        );
    }
    bencher.bench_local(|| {
        // SAFETY: destination holds 2*count limbs and inputs hold count limbs, completely disjoint.
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

#[divan::bench(args = DENSE_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn gmp_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    gmp_serial(bencher, len);
}

// ── 4. FLINT Serial & Parallel Reference ───────────────────────────────────

#[divan::bench(args = DENSE_COMPARE_SIZES)]
fn flint_serial(bencher: divan::Bencher<'_, '_>, len: usize) {
    benchmark_flint(bencher, len, 1);
}

#[divan::bench(args = DENSE_COMPARE_HUGE_SIZES, sample_count = 3, sample_size = 1)]
fn flint_serial_huge(bencher: divan::Bencher<'_, '_>, len: usize) {
    benchmark_flint(bencher, len, 1);
}

#[divan::bench(args = flint_parallel_cases(DENSE_COMPARE_SIZES))]
fn flint_parallel(bencher: divan::Bencher<'_, '_>, case: FlintParallelCase) {
    benchmark_flint(bencher, case.len, case.workers);
}

#[divan::bench(
    args = flint_parallel_cases(DENSE_COMPARE_HUGE_SIZES),
    sample_count = 3,
    sample_size = 1,
)]
fn flint_parallel_huge(bencher: divan::Bencher<'_, '_>, case: FlintParallelCase) {
    benchmark_flint(bencher, case.len, case.workers);
}

fn benchmark_flint(bencher: divan::Bencher<'_, '_>, len: usize, workers: usize) {
    const { assert_one_limb_width() }
    let threads = FlintThreadBudget::new(workers);
    debug_assert_eq!(
        threads.workers(),
        workers,
        "FLINT worker count matches budget"
    );
    let (left, right, mut destination) = operands(len);
    let count = FlintSize::try_from(len).expect("width fits a FLINT size");
    let mut expected = vec![Limb::MIN; left.len().saturating_mul(2)];
    // SAFETY: the probe and input slices are valid non-aliasing spans of len limbs.
    unsafe {
        flint_mpn_mul_n(
            expected.as_mut_ptr().cast::<FlintLimb>(),
            left.as_ptr().cast::<FlintLimb>(),
            right.as_ptr().cast::<FlintLimb>(),
            count,
        );
    }
    bencher.bench_local(|| {
        // SAFETY: destination holds 2*count limbs and inputs hold count limbs, completely disjoint.
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
