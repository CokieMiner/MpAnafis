//! Forced multiplication-algorithm measurements.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{
    Limb, Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{
    KARATSUBA_SIZES, SCHOOLBOOK_SIZES, SSA_SCORECARD_SIZES, TOOM3_SIZES, TOOM4_SIZES, TOOM6_SIZES,
    TOOM8_SIZES, TRANSFORM_SIZES, gmp_equal_reference, operands, validate_and_warm_product,
};

#[divan::bench(args = SCHOOLBOOK_SIZES)]
fn schoolbook(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        Tuner::bench_schoolbook_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = SCHOOLBOOK_SIZES)]
fn schoolbook_raw(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        Tuner::bench_schoolbook_mul_raw(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench]
fn schoolbook_4x4(bencher: divan::Bencher) {
    let (left, right, mut destination) = operands(4);
    bencher.bench_local(|| {
        Tuner::bench_schoolbook_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = KARATSUBA_SIZES)]
fn karatsuba(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; Tuner::bench_karatsuba_mul_scratch_len(len, len)];
    bencher.bench_local(|| {
        Tuner::bench_karatsuba_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM3_SIZES)]
fn toom3_forced(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; Tuner::bench_toom_cook_3_forced_scratch_len(len, len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_3_mul_forced_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM4_SIZES)]
fn toom4_forced(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; Tuner::bench_toom_cook_4_scratch_len(len, len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_4_mul_forced_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM6_SIZES)]
fn toom6_forced(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; Tuner::bench_toom_cook_6_scratch_len(len, len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_6_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TOOM8_SIZES)]
fn toom8_forced(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; Tuner::bench_toom_cook_8_scratch_len(len, len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_8_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

/// Caller-owned-scratch SSA timing: the measured body is pure algorithm.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_forced_end_to_end(bencher: divan::Bencher, len: usize) {
    bench_ssa(
        bencher,
        len,
        SsaGeometryPolicy::Forced,
        SsaScratchPolicy::Reusable,
    );
}

/// Focused caller-owned-scratch cells for SSA/GMP crossover decisions.
#[divan::bench(args = SSA_SCORECARD_SIZES)]
fn ssa_scorecard(bencher: divan::Bencher, len: usize) {
    bench_ssa(
        bencher,
        len,
        SsaGeometryPolicy::Forced,
        SsaScratchPolicy::Reusable,
    );
}

/// Self-allocating SSA timing. The delta against [`ssa_forced_end_to_end`] is
/// the cost of the zeroed scratch allocation, which GMP's `TMP_ALLOC_LIMBS`
/// does not pay.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_end_to_end_allocating(bencher: divan::Bencher, len: usize) {
    bench_ssa(
        bencher,
        len,
        SsaGeometryPolicy::Forced,
        SsaScratchPolicy::Allocating,
    );
}

/// SSA on the exact path production takes: `force_transform` cleared.
///
/// The delta against [`ssa_forced_end_to_end`] is the cost of letting every
/// recursive ring product inside `SSA_BASE_MODULUS_BITS` fall back to the
/// multiplication tower instead of staying in the transform.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_production_end_to_end(bencher: divan::Bencher, len: usize) {
    bench_ssa(
        bencher,
        len,
        SsaGeometryPolicy::Production,
        SsaScratchPolicy::Reusable,
    );
}

fn bench_ssa(
    bencher: divan::Bencher<'_, '_>,
    len: usize,
    geometry: SsaGeometryPolicy,
    scratch: SsaScratchPolicy,
) {
    let (left, right, mut destination) = operands(len);
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        geometry,
        TransformExecutor::Sequential,
        scratch,
        &left,
        &right,
    ) else {
        return;
    };
    let expected = gmp_equal_reference(&left, &right);
    validate_and_warm_product(&expected, "prepared SSA tier product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}
