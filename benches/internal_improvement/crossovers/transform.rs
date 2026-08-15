//! Where the transform should start taking over from the conventional tower.
//!
//! The shape matrix reports its densest remaining deficit just below the
//! transform crossover, so this forces both sides on the same operands: the
//! production dispatcher (which at these widths declines the transform) against
//! the transform invoked directly, plus GMP for scale.
//!
//! `bench_ssa_mul_with_scratch` ignores the crossover, which is the whole point
//! -- the question is what the crossover *should* be, and that cannot be asked
//! through a dispatcher that already answers it.
//!
//! Three arms, not two. Above `SSA_THRESHOLD` the dispatcher already selects the
//! transform, so `tower` and `forced_transform` measure the same path there and
//! the comparison goes blind exactly where the crossover sits. `forced_toom8`
//! keeps the conventional tower reachable across the whole ladder, so both sides
//! stay measurable on either side of the threshold.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    algorithms::{bench_toom_cook_8_mul_with_scratch, bench_toom_cook_8_scratch_len},
    state::bench_mul_tower_pooled,
    transform::{bench_ssa_mul_scratch_len, bench_ssa_mul_with_scratch},
};

use crate::shared::operands_pair;

/// Balanced widths straddling `SSA_THRESHOLD`, at a finer step than the shape
/// ladder below. The `compare::balanced` run has the transform behind GMP at
/// 4096 limbs and ahead of it at 8192, so the threshold's own neighbourhood is
/// what needs resolving.
const CROSSOVER_WIDTHS: [usize; 13] = [
    2560, 2624, 2688, 2752, 2816, 3072, 3328, 3584, 4096, 4608, 5120, 5632, 6144,
];

const SHAPES: [(usize, usize); 24] = [
    (2000, 2000),
    (2000, 1600),
    (2000, 1333),
    (2500, 2500),
    (2500, 2000),
    (2500, 1666),
    (3000, 3000),
    (3000, 2400),
    (3000, 2000),
    (3500, 3500),
    (3500, 2800),
    (3500, 2333),
    (4091, 4091),
    (4091, 3272),
    (4091, 2727),
    (4600, 4600),
    (4600, 3680),
    (4600, 3066),
    (5200, 5200),
    (5200, 4160),
    (5200, 3466),
    (6000, 6000),
    (6000, 4800),
    (6000, 4000),
];

#[divan::bench(args = SHAPES)]
fn tower(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    bencher.bench_local(|| {
        bench_mul_tower_pooled(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = SHAPES)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(larger_len, smaller_len)];
    bencher.bench_local(|| {
        bench_ssa_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = SHAPES)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let larger_count = size_t::try_from(larger_len).expect("width fits a GMP size");
    let smaller_count = size_t::try_from(smaller_len).expect("width fits a GMP size");
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

// ── Balanced probe across the threshold ──────────────────────────────────────

/// What production actually runs, so the forced arms below are read against the
/// dispatcher's own choice rather than against each other alone.
#[divan::bench(args = CROSSOVER_WIDTHS)]
fn balanced_tower(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    bencher.bench_local(|| {
        bench_mul_tower_pooled(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

#[divan::bench(args = CROSSOVER_WIDTHS)]
fn balanced_forced_transform(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    let mut scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_ssa_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = CROSSOVER_WIDTHS)]
fn balanced_forced_toom8(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_8_scratch_len(len, len)];
    bencher.bench_local(|| {
        bench_toom_cook_8_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = CROSSOVER_WIDTHS)]
fn balanced_gmp(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (larger, smaller, mut destination) = operands_pair(len, len);
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: three independently allocated, disjoint vectors of exactly
        // `count` limbs each, with a destination sized for the full product.
        let _high = unsafe {
            gmp::mpn_mul(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(larger.as_ptr().cast::<limb_t>()),
                black_box(count),
                black_box(smaller.as_ptr().cast::<limb_t>()),
                black_box(count),
            )
        };
        let _output = black_box(&destination);
    });
}
