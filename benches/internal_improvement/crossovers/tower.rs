//! Local crossover windows and configured-tower measurements.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{Limb, Tuner, state::MultiplicationBenchState};

use crate::shared::{
    BASECASE_CROSSOVER_SIZES, BASECASE_ROW_SIZES, BASECASE_WIDTH_SIZES, LOWER_CHILD_SIZES,
    LOWER_TOWER_CROSSOVER_SIZES, TOWER_SIZES, operands, operands_pair,
};

/// Square schoolbook products across every width the ADX backend specializes.
///
/// `mul_basecase_unchecked` dispatches its fixed-width kernels on the *inner*
/// operand width, so this sweep is what tells the fixed-width table apart from
/// the general-length ADX loop.
#[divan::bench(args = BASECASE_WIDTH_SIZES)]
fn basecase_width_schoolbook(bencher: divan::Bencher, len: usize) {
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

/// Four limbs against a long row: the rectangular shape the tower produces
/// whenever a Karatsuba split would leave an empty half.
#[divan::bench(args = BASECASE_ROW_SIZES)]
fn basecase_row_schoolbook(bencher: divan::Bencher, row: usize) {
    let (left, right, mut destination) = operands_pair(4, row);
    bencher.bench_local(|| {
        Tuner::bench_schoolbook_mul(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = BASECASE_CROSSOVER_SIZES)]
fn basecase_crossover_schoolbook(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = BASECASE_CROSSOVER_SIZES)]
fn basecase_crossover_karatsuba(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = LOWER_CHILD_SIZES)]
fn lower_child_karatsuba(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = LOWER_TOWER_CROSSOVER_SIZES)]
fn lower_crossover_karatsuba(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = LOWER_TOWER_CROSSOVER_SIZES)]
fn lower_crossover_toom3(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = LOWER_TOWER_CROSSOVER_SIZES)]
fn lower_crossover_toom4(bencher: divan::Bencher, len: usize) {
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

#[divan::bench(args = TOWER_SIZES)]
fn mp_tower_reused(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    let mut scratch = MultiplicationBenchState::default();
    let mut prepared = scratch.prepare(&mut destination, &left, &right);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}

#[divan::bench(args = TOWER_SIZES)]
fn mp_tower_pooled(bencher: divan::Bencher, len: usize) {
    let (left, right, mut destination) = operands(len);
    bencher.bench_local(|| {
        Tuner::bench_mul_tower_pooled(
            black_box(&mut destination),
            black_box(&left),
            black_box(&right),
        );
        let _output = black_box(&destination);
    });
}
