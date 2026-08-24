//! Which transform geometry the planner should pick just above the crossover.
//!
//! The shape matrix leaves one unexplained cell: a 10000 by 7500 product runs
//! 1.12x behind the reference while both of its ratio neighbours are ahead of
//! it. Every one of those shapes selects the same tier, so the tier is not the
//! variable -- the transform *geometry* is. This forces each exponent in the
//! planner's search window on the same operands and reports what the planner
//! chose alongside them.
//!
//! An exponent of zero means the production planner picks. Exponents the ring
//! cannot support are skipped, and report as an empty timing rather than
//! panicking, because the valid window depends on the ring width and so differs
//! between the shapes compared here.

#![expect(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{gmp_pair_reference, operands_pair, validate_and_warm_product};

const SSA_2048_EXPONENTS: [u32; 4] = [7, 8, 9, 10];
const SSA_4096_EXPONENTS: [u32; 5] = [6, 7, 8, 9, 10];
const SSA_8192_EXPONENTS: [u32; 5] = [6, 7, 8, 9, 10];
const SSA_16384_EXPONENTS: [u32; 5] = [7, 8, 9, 10, 11];
/// Transform-exponent sweep spanning the planner's analytic centre around
/// every larger size where the inner ring changes regime. Widths with a
/// dedicated row above are deliberately absent, so every `(width, exponent)`
/// cell is timed once.
const SSA_LARGE_GEOMETRIES: [(usize, u32); 55] = [
    (32_768, 8),
    (32_768, 9),
    (32_768, 10),
    (32_768, 11),
    (32_768, 12),
    (65_536, 8),
    (65_536, 9),
    (65_536, 10),
    (65_536, 11),
    (65_536, 12),
    (98_304, 9),
    (98_304, 10),
    (98_304, 11),
    (131_072, 9),
    (131_072, 10),
    (131_072, 11),
    (131_072, 12),
    (131_072, 13),
    (196_608, 10),
    (196_608, 11),
    (262_144, 10),
    (262_144, 11),
    (262_144, 12),
    (262_144, 13),
    (524_288, 11),
    (524_288, 12),
    (524_288, 13),
    (524_288, 14),
    (524_288, 15),
    (1_048_576, 11),
    (1_048_576, 12),
    (1_048_576, 13),
    (1_048_576, 14),
    (1_048_576, 15),
    (1_048_576, 16),
    (2_097_152, 12),
    (2_097_152, 13),
    (2_097_152, 14),
    (2_097_152, 15),
    (2_097_152, 16),
    (2_097_152, 17),
    (4_194_304, 12),
    (4_194_304, 13),
    (4_194_304, 14),
    (4_194_304, 15),
    (4_194_304, 16),
    (4_194_304, 17),
    (4_194_304, 18),
    (8_388_608, 13),
    (8_388_608, 14),
    (8_388_608, 15),
    (8_388_608, 16),
    (8_388_608, 17),
    (8_388_608, 18),
    (8_388_608, 19),
];
const PROBES: [(usize, usize, u32); 40] = [
    (10000, 7500, 0),
    (10000, 7500, 6),
    (10000, 7500, 7),
    (10000, 7500, 8),
    (10000, 7500, 9),
    (10000, 7500, 10),
    (10000, 7500, 11),
    (10000, 7500, 12),
    (10000, 7500, 13),
    (10000, 7500, 14),
    (10000, 8000, 0),
    (10000, 8000, 6),
    (10000, 8000, 7),
    (10000, 8000, 8),
    (10000, 8000, 9),
    (10000, 8000, 10),
    (10000, 8000, 11),
    (10000, 8000, 12),
    (10000, 8000, 13),
    (10000, 8000, 14),
    (10000, 6666, 0),
    (10000, 6666, 6),
    (10000, 6666, 7),
    (10000, 6666, 8),
    (10000, 6666, 9),
    (10000, 6666, 10),
    (10000, 6666, 11),
    (10000, 6666, 12),
    (10000, 6666, 13),
    (10000, 6666, 14),
    (10000, 10000, 0),
    (10000, 10000, 6),
    (10000, 10000, 7),
    (10000, 10000, 8),
    (10000, 10000, 9),
    (10000, 10000, 10),
    (10000, 10000, 11),
    (10000, 10000, 12),
    (10000, 10000, 13),
    (10000, 10000, 14),
];

#[divan::bench(args = SSA_2048_EXPONENTS)]
fn balanced_2048(bencher: divan::Bencher, exponent: u32) {
    bench_balanced_geometry(bencher, 2_048, exponent);
}

#[divan::bench(args = SSA_4096_EXPONENTS)]
fn balanced_4096(bencher: divan::Bencher, exponent: u32) {
    bench_balanced_geometry(bencher, 4_096, exponent);
}

#[divan::bench(args = SSA_8192_EXPONENTS)]
fn balanced_8192(bencher: divan::Bencher, exponent: u32) {
    bench_balanced_geometry(bencher, 8_192, exponent);
}

#[divan::bench(args = SSA_16384_EXPONENTS)]
fn balanced_16384(bencher: divan::Bencher, exponent: u32) {
    bench_balanced_geometry(bencher, 16_384, exponent);
}

#[divan::bench(args = SSA_LARGE_GEOMETRIES)]
fn balanced_large(bencher: divan::Bencher, plan: (usize, u32)) {
    let (len, exponent) = plan;
    bench_balanced_geometry(bencher, len, exponent);
}

#[divan::bench(args = PROBES)]
fn unbalanced(bencher: divan::Bencher<'_, '_>, probe: (usize, usize, u32)) {
    let (larger_len, smaller_len, exponent) = probe;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let geometry = if exponent == 0 {
        SsaGeometryPolicy::Production
    } else {
        SsaGeometryPolicy::ForcedExponent(exponent)
    };
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        geometry,
        TransformExecutor::Sequential,
        SsaScratchPolicy::Reusable,
        &larger,
        &smaller,
    ) else {
        return;
    };
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "SSA geometry product", |candidate| {
        runner.prepare(candidate).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

#[divan::bench(args = PROBES)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, probe: (usize, usize, u32)) {
    let (larger_len, smaller_len, exponent) = probe;
    if exponent != 0 {
        return;
    }
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count_result = size_t::try_from(larger_len);
    let smaller_count_result = size_t::try_from(smaller_len);
    assert!(
        larger_count_result.is_ok() && smaller_count_result.is_ok(),
        "benchmark widths must fit GMP sizes"
    );
    // SAFETY: the assertion immediately above validates both conversions once,
    // before the timed loop.
    let (larger_count, smaller_count) = unsafe {
        (
            larger_count_result.unwrap_unchecked(),
            smaller_count_result.unwrap_unchecked(),
        )
    };
    bencher.bench_local(|| {
        // SAFETY: three independently allocated, disjoint vectors of exactly the
        // stated limb counts, longer operand first as mpn_mul requires.
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

fn bench_balanced_geometry(bencher: divan::Bencher<'_, '_>, len: usize, exponent: u32) {
    let (left, right, mut destination) = operands_pair(len, len);
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        SsaGeometryPolicy::ForcedExponent(exponent),
        TransformExecutor::Sequential,
        SsaScratchPolicy::Reusable,
        &left,
        &right,
    ) else {
        return;
    };
    let expected = gmp_pair_reference(&left, &right);
    validate_and_warm_product(&expected, "balanced SSA geometry product", |candidate| {
        runner.prepare(candidate).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}
