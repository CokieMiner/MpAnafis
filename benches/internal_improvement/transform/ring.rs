//! SSA ring choices and their leaf operations.
//!
//! `SSA_BASE_MODULUS_BITS` is the widest inner ring handled by the
//! multiplication tower; above it the pointwise stage nests another transform.
//! It was last swept before the inner-ring rounding fix, which cut the cost of
//! being on the nested side by up to a factor of two, so the crossover it
//! encodes almost certainly moved.
//!
//! This forces the transform across balanced widths spanning the range where the
//! nested branch is reachable. It is meant to be run once per candidate value
//! through `MP_TUNING_PROFILE`, because the constant is compiled in.

use core::{hint::black_box, mem::size_of};

use mp_anafis::tune_api::tier::{
    Limb, Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{
    FERMAT_TRANSFORM_SIZES, gmp_pair_reference, operand, operands, operands_pair,
    validate_and_warm_product,
};

const WIDTHS: [usize; 6] = [8_192, 32_769, 131_073, 262_145, 524_289, 1_048_577];
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

#[divan::bench(args = WIDTHS)]
fn base_modulus_transition(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        SsaGeometryPolicy::Forced,
        TransformExecutor::Sequential,
        SsaScratchPolicy::Reusable,
        &larger,
        &smaller,
    ) else {
        return;
    };
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "forced SSA base-modulus product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

#[divan::bench(args = FERMAT_TRANSFORM_SIZES)]
fn fermat_full_width(bencher: divan::Bencher, len: usize) {
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len.wrapping_add(1)];
    bencher.bench_local(|| {
        Tuner::bench_ssa_fermat_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = FERMAT_TRANSFORM_SIZES)]
fn mersenne_full_width(bencher: divan::Bencher, len: usize) {
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len];
    let mut scratch = vec![Limb::MIN; Tuner::bench_ssa_mersenne_mul_scratch_len(len)];
    bencher.bench_local(|| {
        Tuner::bench_ssa_mersenne_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = [65_usize, 128, 256, 1_024])]
fn shift_from_negated(bencher: divan::Bencher, len: usize) {
    let mut source = operand(len.wrapping_add(1), Limb::MAX.wrapping_sub(0x1234));
    let Some(coefficient_top) = source.last_mut() else {
        return;
    };
    *coefficient_top = 0;
    let mut destination = vec![Limb::MIN; source.len()];
    let shift = len
        .wrapping_mul(size_of::<Limb>().wrapping_mul(8))
        .wrapping_add(31);
    Tuner::bench_ssa_shift_from(&mut destination, &source, shift);
    bencher.bench_local(|| {
        Tuner::bench_ssa_shift_from(
            black_box(&mut destination),
            black_box(&source),
            black_box(shift),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = FERMAT_GEOMETRIES)]
fn fermat_geometry(bencher: divan::Bencher, plan: (usize, u32)) {
    let (len, transform_exponent) = plan;
    let (left, right, _) = operands(len);
    let mut destination = vec![Limb::MIN; len.wrapping_add(1)];
    let mut expected = vec![Limb::MIN; len.wrapping_add(1)];
    Tuner::bench_ssa_fermat_mul(&mut expected, &left, &right);
    Tuner::bench_ssa_fermat_mul_forced_plan(&mut destination, &left, &right, transform_exponent);
    assert_eq!(
        destination, expected,
        "forced Fermat geometry disagrees with the default ring product"
    );
    bencher.bench_local(|| {
        Tuner::bench_ssa_fermat_mul_forced_plan(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
            black_box(transform_exponent),
        );
        let _output = black_box(&destination);
    });
}
