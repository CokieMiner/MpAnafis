//! Forced multiplication-algorithm measurements.

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{
    Limb,
    algorithms::{
        bench_karatsuba_mul_scratch_len, bench_karatsuba_mul_with_scratch, bench_schoolbook_mul,
        bench_schoolbook_mul_raw, bench_toom_cook_3_forced_scratch_len,
        bench_toom_cook_3_mul_forced_with_scratch, bench_toom_cook_4_mul_forced_with_scratch,
        bench_toom_cook_4_scratch_len, bench_toom_cook_6_mul_with_scratch,
        bench_toom_cook_6_scratch_len, bench_toom_cook_8_mul_with_scratch,
        bench_toom_cook_8_scratch_len,
    },
    transform::{
        bench_ntt_mul, bench_ntt_mul_forced, bench_ssa_fermat_mul,
        bench_ssa_fermat_mul_forced_plan, bench_ssa_mersenne_mul,
        bench_ssa_mersenne_mul_scratch_len, bench_ssa_mul, bench_ssa_mul_forced_plan,
        bench_ssa_mul_forced_plan_scratch_len, bench_ssa_mul_production, bench_ssa_mul_scratch_len,
        bench_ssa_mul_with_scratch,
    },
};

const SSA_2048_EXPONENTS: [u32; 4] = [7, 8, 9, 10];
const SSA_4096_EXPONENTS: [u32; 5] = [6, 7, 8, 9, 10];
const SSA_8192_EXPONENTS: [u32; 5] = [6, 7, 8, 9, 10];
const SSA_16384_EXPONENTS: [u32; 5] = [7, 8, 9, 10, 11];
/// Transform-exponent sweep spanning the planner's analytic centre ±2 at every
/// size where the inner ring crosses `SSA_BASE_MODULUS_BITS`. This is the
/// evidence base for the cost model in `ssa/plan.rs`.
const SSA_LARGE_GEOMETRIES: [(usize, u32); 66] = [
    (4_096, 6),
    (4_096, 7),
    (4_096, 8),
    (4_096, 9),
    (4_096, 10),
    (16_384, 7),
    (16_384, 8),
    (16_384, 9),
    (16_384, 10),
    (16_384, 11),
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
    (131_072, 9),
    (131_072, 10),
    (131_072, 11),
    (131_072, 12),
    (131_072, 13),
    (262_144, 10),
    (262_144, 11),
    (262_144, 12),
    (262_144, 13),
    (262_144, 14),
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
    (98_304, 9),
    (98_304, 10),
    (98_304, 11),
    (196_608, 10),
    (196_608, 11),
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
const FERMAT_GEOMETRIES: [(usize, u32); 17] = [
    (4_096, 7),
    (4_096, 8),
    (4_096, 9),
    (4_096, 10),
    (8_192, 8),
    (8_192, 9),
    (8_192, 10),
    (16_384, 8),
    (16_384, 9),
    (16_384, 10),
    (16_384, 11),
    (524_288, 10),
    (524_288, 11),
    (524_288, 12),
    (1_048_576, 11),
    (1_048_576, 12),
    (1_048_576, 13),
];

use crate::shared::{
    FERMAT_TRANSFORM_SIZES, GOLDILOCKS_23_SIZES, KARATSUBA_SIZES, SCHOOLBOOK_SIZES,
    SSA_SCORECARD_SIZES, TOOM3_SIZES, TOOM4_SIZES, TOOM6_SIZES, TOOM8_SIZES, TRANSFORM_SIZES,
    TWO_PRIME_19_SIZES, TWO_PRIME_20_SIZES, operand, operands,
};

#[divan::bench(args = SCHOOLBOOK_SIZES)]
fn schoolbook(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_schoolbook_mul(
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
        bench_schoolbook_mul_raw(
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
        bench_schoolbook_mul(
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
    let mut scratch = vec![Limb::MIN; bench_karatsuba_mul_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_karatsuba_mul_with_scratch(
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
    let mut scratch = vec![Limb::MIN; bench_toom_cook_3_forced_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_toom_cook_3_mul_forced_with_scratch(
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
    let mut scratch = vec![Limb::MIN; bench_toom_cook_4_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_toom_cook_4_mul_forced_with_scratch(
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
    let mut scratch = vec![Limb::MIN; bench_toom_cook_6_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_toom_cook_6_mul_with_scratch(
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
    let mut scratch = vec![Limb::MIN; bench_toom_cook_8_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_toom_cook_8_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TRANSFORM_SIZES)]
fn ntt_forced_end_to_end(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ntt_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = GOLDILOCKS_23_SIZES)]
fn ntt_goldilocks_23(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ntt_mul_forced(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            23,
            1,
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TWO_PRIME_20_SIZES)]
fn ntt_two_prime_20(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ntt_mul_forced(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            20,
            2,
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TWO_PRIME_19_SIZES)]
fn ntt_two_prime_19(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ntt_mul_forced(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            19,
            2,
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = TRANSFORM_SIZES)]
fn ntt_three_prime_31(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ntt_mul_forced(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            31,
            3,
        );
        let _output = black_box(&destination);
    });
}

/// Caller-owned-scratch SSA timing: the measured body is pure algorithm.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_forced_end_to_end(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_ssa_mul_with_scratch(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

/// Focused caller-owned-scratch cells for SSA/GMP crossover decisions.
#[divan::bench(args = SSA_SCORECARD_SIZES)]
fn ssa_scorecard(bencher: divan::Bencher, len: usize) {
    ssa_forced_end_to_end(bencher, len);
}

/// Self-allocating SSA timing. The delta against [`ssa_forced_end_to_end`] is
/// the cost of the zeroed scratch allocation, which GMP's `TMP_ALLOC_LIMBS`
/// does not pay.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_end_to_end_allocating(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        bench_ssa_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

/// SSA on the exact path production takes: `force_transform` cleared.
///
/// The delta against [`ssa_forced_end_to_end`] is the cost of letting every
/// recursive ring product inside `SSA_BASE_MODULUS_BITS` fall back to the
/// multiplication tower instead of staying in the transform.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_production_end_to_end(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    // The tower hands SSA a `set_len` buffer whose contents are arbitrary, so
    // this one is poisoned rather than zeroed to match. Measured identical to a
    // zeroed buffer, which rules scratch contents out as a source of the gap
    // between this benchmark and `crossovers::arbi_tower_reused`.
    let mut scratch = operand(
        bench_ssa_mul_scratch_len(len, len),
        Limb::MAX.wrapping_sub(0x5555),
    );
    bencher.bench_local(|| {
        bench_ssa_mul_production(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

/// Cost of *sizing* an SSA product, with no multiplication at all.
///
/// `ssa_mul_scratch_len` walks `crt_layout_len` -> `mul_mod_bnm1_scratch_len`,
/// which halves the width down to the `B^n - 1` basecase and runs the full
/// `FftPlan` cost model at every level. The tower pays this once in
/// `mul_plan_scratch_len` and `ssa_mul_into` pays it again internally, so it is
/// charged at least twice per product. Anything non-trivial here is pure
/// overhead against GMP, which resolves its FFT geometry from a table.
#[divan::bench(args = TRANSFORM_SIZES)]
fn ssa_scratch_len_planning(bencher: divan::Bencher, len: usize) {
    bencher.bench_local(|| black_box(bench_ssa_mul_scratch_len(black_box(len), black_box(len))));
}

#[divan::bench(args = SSA_2048_EXPONENTS)]
fn ssa_2048_geometry(bencher: divan::Bencher, transform_exponent: u32) {
    bench_forced_geometry(bencher, 2_048, transform_exponent);
}

#[divan::bench(args = SSA_4096_EXPONENTS)]
fn ssa_4096_geometry(bencher: divan::Bencher, transform_exponent: u32) {
    bench_forced_geometry(bencher, 4_096, transform_exponent);
}

#[divan::bench(args = SSA_8192_EXPONENTS)]
fn ssa_8192_geometry(bencher: divan::Bencher, transform_exponent: u32) {
    bench_forced_geometry(bencher, 8_192, transform_exponent);
}

#[divan::bench(args = SSA_16384_EXPONENTS)]
fn ssa_16384_geometry(bencher: divan::Bencher, transform_exponent: u32) {
    bench_forced_geometry(bencher, 16_384, transform_exponent);
}

#[divan::bench(args = SSA_LARGE_GEOMETRIES)]
fn ssa_large_geometry(bencher: divan::Bencher, plan: (usize, u32)) {
    let (len, transform_exponent) = plan;
    bench_forced_geometry(bencher, len, transform_exponent);
}

/// Times one forced SSA geometry with the exact caller-owned scratch, after
/// cross-checking the forced plan against the planner's own choice.
fn bench_forced_geometry(bencher: divan::Bencher, len: usize, transform_exponent: u32) {
    let (left, right, mut destination) = operands(len);
    let scratch_len = bench_ssa_mul_forced_plan_scratch_len(len, len, transform_exponent)
        .expect("configured forced SSA geometry is valid for this operand size");
    let mut scratch = vec![Limb::MIN; scratch_len];

    let mut expected = vec![Limb::MIN; destination.len()];
    let mut default_scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(len, len)];
    bench_ssa_mul_with_scratch(&mut expected, &left, &right, &mut default_scratch);
    drop(default_scratch);
    bench_ssa_mul_forced_plan(
        &mut destination,
        &left,
        &right,
        transform_exponent,
        &mut scratch,
    );
    assert_eq!(
        destination, expected,
        "forced SSA geometry disagrees with the planner's default geometry"
    );
    drop(expected);

    bencher.bench_local(|| {
        bench_ssa_mul_forced_plan(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(transform_exponent),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = FERMAT_TRANSFORM_SIZES)]
fn ssa_fermat_full_width(bencher: divan::Bencher, len: usize) {
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len.wrapping_add(1)];
    bencher.bench_local(|| {
        bench_ssa_fermat_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = FERMAT_TRANSFORM_SIZES)]
fn ssa_mersenne_full_width(bencher: divan::Bencher, len: usize) {
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len];
    let mut scratch = vec![Limb::MIN; bench_ssa_mersenne_mul_scratch_len(len)];
    bencher.bench_local(|| {
        bench_ssa_mersenne_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = FERMAT_GEOMETRIES)]
fn ssa_fermat_geometry(bencher: divan::Bencher, plan: (usize, u32)) {
    let (len, transform_exponent) = plan;
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len.wrapping_add(1)];
    let mut expected = vec![Limb::MIN; len.wrapping_add(1)];
    bench_ssa_fermat_mul(&mut expected, &left, &right);
    bench_ssa_fermat_mul_forced_plan(&mut destination, &left, &right, transform_exponent);
    assert_eq!(
        destination, expected,
        "forced Fermat geometry disagrees with the default ring product"
    );
    bencher.bench_local(|| {
        bench_ssa_fermat_mul_forced_plan(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(transform_exponent),
        );
        let _output = black_box(&destination);
    });
}
