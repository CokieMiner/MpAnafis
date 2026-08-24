//! Fractional-ratio Toom against the blocked path it displaces.
//!
//! Every shape here is one the blocked path would otherwise take:
//! `prefers_blocked_product` fires from four-to-three upward, so the whole band
//! the fractional tiers admit was previously cut into blocks. The comparison is
//! therefore the decision itself -- not the tier against a balanced ladder that
//! never sees these operands.
//!
//! Two shape sets, because the tiers do not cover the same band. `OWN_SHAPES`
//! are ratios only the four-by-three split can express; `SHARED_SHAPES` are the
//! `[1.5, 2)` overlap where both admit the shape and the selector has to order
//! them.
//!
//! The blocked arm uses the production block width so the arms differ only in
//! the algorithm, and all arms take caller-owned scratch so none pays an
//! allocation the others avoid.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{Limb, Tuner};

use crate::shared::operands_pair;

/// Ratios in `[4/3, 1.5)`, which only the four-by-three split can express.
const OWN_SHAPES: [(usize, usize); 18] = [
    (192, 144),
    (192, 137),
    (192, 134),
    (384, 288),
    (384, 274),
    (384, 268),
    (768, 576),
    (768, 548),
    (768, 537),
    (1536, 1152),
    (1536, 1097),
    (1536, 1075),
    (3072, 2304),
    (3072, 2194),
    (3072, 2150),
    (4608, 3456),
    (4608, 3291),
    (4608, 3225),
];

/// The `[1.5, 2)` overlap both fractional tiers admit.
const SHARED_SHAPES: [(usize, usize); 18] = [
    (192, 128),
    (192, 120),
    (192, 115),
    (384, 256),
    (384, 240),
    (384, 230),
    (768, 512),
    (768, 480),
    (768, 460),
    (1536, 1024),
    (1536, 960),
    (1536, 921),
    (3072, 2048),
    (3072, 1920),
    (3072, 1843),
    (4608, 3072),
    (4608, 2880),
    (4608, 2764),
];

#[divan::bench(args = OWN_SHAPES)]
fn own_blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    blocked_arm(bencher, shape);
}

#[divan::bench(args = OWN_SHAPES)]
fn own_toom43(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    toom43_arm(bencher, shape);
}

#[divan::bench(args = SHARED_SHAPES)]
fn shared_blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    blocked_arm(bencher, shape);
}

#[divan::bench(args = SHARED_SHAPES)]
fn shared_toom32(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch =
        vec![Limb::MIN; Tuner::bench_toom_cook_32_scratch_len(larger_len, smaller_len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_32_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = SHARED_SHAPES)]
fn shared_toom43(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    toom43_arm(bencher, shape);
}

fn toom43_arm(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch =
        vec![Limb::MIN; Tuner::bench_toom_cook_43_scratch_len(larger_len, smaller_len)];
    bencher.bench_local(|| {
        Tuner::bench_toom_cook_43_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(&mut scratch),
        );
    });
}

fn blocked_arm(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let block_len = smaller_len;
    let mut scratch =
        vec![Limb::MIN; Tuner::bench_lopsided_mul_scratch_len(larger_len, smaller_len, block_len)];
    bencher.bench_local(|| {
        Tuner::bench_lopsided_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(block_len),
            black_box(&mut scratch),
        );
    });
}
