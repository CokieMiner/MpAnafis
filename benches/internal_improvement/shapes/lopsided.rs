//! Production-tower comparisons for highly unbalanced multiplication.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::{hint::black_box, mem::size_of};

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    state::{MulBenchScratch, bench_lopsided_mul_scratch_len, bench_lopsided_mul_with_scratch},
};

use crate::shared::operands_pair;

const LOPSIDED_SHAPES: [(usize, usize); 18] = [
    (24, 8),
    (24, 16),
    (24, 64),
    (64, 8),
    (64, 16),
    (64, 64),
    (128, 8),
    (128, 16),
    (128, 64),
    (192, 8),
    (192, 16),
    (192, 64),
    (384, 8),
    (384, 16),
    (384, 64),
    (1_024, 8),
    (1_024, 16),
    (1_024, 64),
];

const GEOMETRY_SHAPES: [(usize, usize, usize, usize); 75] = [
    (128, 8, 1, 2),
    (128, 8, 2, 3),
    (128, 8, 3, 4),
    (128, 8, 7, 8),
    (128, 8, 1, 1),
    (128, 8, 9, 8),
    (128, 8, 7, 6),
    (128, 8, 5, 4),
    (128, 16, 1, 2),
    (128, 16, 2, 3),
    (128, 16, 3, 4),
    (128, 16, 7, 8),
    (128, 16, 1, 1),
    (128, 16, 9, 8),
    (128, 16, 7, 6),
    (128, 16, 5, 4),
    (128, 64, 1, 2),
    (128, 64, 2, 3),
    (128, 64, 3, 4),
    (128, 64, 7, 8),
    (128, 64, 1, 1),
    (128, 64, 9, 8),
    (128, 64, 7, 6),
    (128, 64, 5, 4),
    (192, 8, 1, 2),
    (192, 8, 2, 3),
    (192, 8, 3, 4),
    (192, 8, 7, 8),
    (192, 8, 1, 1),
    (192, 8, 9, 8),
    (192, 8, 7, 6),
    (192, 8, 5, 4),
    (192, 16, 1, 2),
    (192, 16, 2, 3),
    (192, 16, 3, 4),
    (192, 16, 7, 8),
    (192, 16, 1, 1),
    (192, 16, 9, 8),
    (192, 16, 7, 6),
    (192, 16, 5, 4),
    (192, 64, 1, 2),
    (192, 64, 2, 3),
    (192, 64, 3, 4),
    (192, 64, 7, 8),
    (192, 64, 1, 1),
    (192, 64, 9, 8),
    (192, 64, 7, 6),
    (192, 64, 5, 4),
    (1_024, 8, 1, 2),
    (1_024, 8, 2, 3),
    (1_024, 8, 3, 4),
    (1_024, 8, 7, 8),
    (1_024, 8, 1, 1),
    (1_024, 8, 9, 8),
    (1_024, 8, 7, 6),
    (1_024, 8, 5, 4),
    (1_024, 8, 8, 7),
    (1_024, 16, 1, 2),
    (1_024, 16, 2, 3),
    (1_024, 16, 3, 4),
    (1_024, 16, 7, 8),
    (1_024, 16, 1, 1),
    (1_024, 16, 9, 8),
    (1_024, 16, 7, 6),
    (1_024, 16, 5, 4),
    (1_024, 16, 16, 15),
    (1_024, 64, 1, 2),
    (1_024, 64, 2, 3),
    (1_024, 64, 3, 4),
    (1_024, 64, 7, 8),
    (1_024, 64, 1, 1),
    (1_024, 64, 9, 8),
    (1_024, 64, 7, 6),
    (1_024, 64, 5, 4),
    (1_024, 64, 64, 57),
];

#[divan::bench(args = LOPSIDED_SHAPES)]
fn mp_production(bencher: divan::Bencher, shape: (usize, usize)) {
    let (smaller_len, ratio) = shape;
    let larger_len = smaller_len
        .checked_mul(ratio)
        .expect("configured lopsided benchmark width fits usize");
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch = MulBenchScratch::default();

    bencher.bench_local(|| {
        scratch.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = GEOMETRY_SHAPES)]
fn mp_forced_geometry(bencher: divan::Bencher, shape: (usize, usize, usize, usize)) {
    let (smaller_len, ratio, block_numerator, block_denominator) = shape;
    let larger_len = smaller_len
        .checked_mul(ratio)
        .expect("configured lopsided benchmark width fits usize");
    let block_len = smaller_len
        .saturating_mul(block_numerator)
        .div_ceil(block_denominator);
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let scratch_len = bench_lopsided_mul_scratch_len(larger_len, smaller_len, block_len);
    let mut scratch = vec![Limb::MIN; scratch_len];

    bencher.bench_local(|| {
        bench_lopsided_mul_with_scratch(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(block_len),
            black_box(&mut scratch),
        );
        let _output = black_box(&destination);
    });
}

#[divan::bench(args = LOPSIDED_SHAPES)]
fn gmp_production(bencher: divan::Bencher, shape: (usize, usize)) {
    assert_eq!(
        size_of::<Limb>(),
        size_of::<limb_t>(),
        "benchmark requires equal Mp and GMP limb widths"
    );
    let (smaller_len, ratio) = shape;
    let larger_len = smaller_len
        .checked_mul(ratio)
        .expect("configured lopsided benchmark width fits usize");
    let (mp_larger, mp_smaller, _) = operands_pair(larger_len, smaller_len);
    let larger: Vec<limb_t> = mp_larger
        .into_iter()
        .map(|limb| limb_t::try_from(limb).expect("Mp limb fits GMP limb"))
        .collect();
    let smaller: Vec<limb_t> = mp_smaller
        .into_iter()
        .map(|limb| limb_t::try_from(limb).expect("Mp limb fits GMP limb"))
        .collect();
    let destination_len = larger_len
        .checked_add(smaller_len)
        .expect("configured lopsided product width fits usize");
    let mut destination = vec![limb_t::MIN; destination_len];
    let gmp_larger_len = size_t::try_from(larger_len).expect("larger width fits GMP mp_size_t");
    let gmp_smaller_len = size_t::try_from(smaller_len).expect("smaller width fits GMP mp_size_t");

    bencher.bench_local(|| {
        // SAFETY: GMP requires the first operand to be at least as long as the
        // second. The ratio is at least eight, all vectors are independently
        // allocated, and destination has larger_len+smaller_len writable limbs.
        unsafe {
            let _high_limb = gmp::mpn_mul(
                black_box(destination.as_mut_ptr()),
                black_box(larger.as_ptr()),
                black_box(gmp_larger_len),
                black_box(smaller.as_ptr()),
                black_box(gmp_smaller_len),
            );
        }
        let _output = black_box(&destination);
    });
}
