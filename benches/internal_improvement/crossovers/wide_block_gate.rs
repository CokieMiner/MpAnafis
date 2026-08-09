//! Transform against production blocking, after the block width was widened.
//!
//! `transform_padding_is_affordable` was derived when `lopsided_block_len` still
//! capped every block just above the shorter operand. That made blocking look
//! far worse than it is at extreme ratios, so the gate sent shapes to a single
//! whole-product transform whose ring is sized by the longer operand while only
//! `smaller` limbs of content exist.
//!
//! With the wide-block rule the comparison has to be made again, and this time
//! against the block width production actually picks rather than a forced one.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    state::{bench_lopsided_mul_production, bench_lopsided_mul_production_scratch_len},
    transform::{bench_ssa_mul_scratch_len, bench_ssa_mul_with_scratch},
};

use crate::shared::operands_pair;

const SHAPES: [(usize, usize); 36] = [
    (16_385, 2048),
    (16_385, 1024),
    (16_385, 512),
    (16_385, 256),
    (16_385, 128),
    (16_385, 64),
    (32_769, 4096),
    (32_769, 2048),
    (32_769, 1024),
    (32_769, 512),
    (32_769, 256),
    (32_769, 128),
    (65_537, 8192),
    (65_537, 4096),
    (65_537, 2048),
    (65_537, 1024),
    (65_537, 512),
    (65_537, 256),
    (131_073, 16_384),
    (131_073, 8192),
    (131_073, 4096),
    (131_073, 2048),
    (131_073, 1024),
    (131_073, 512),
    (262_145, 32_768),
    (262_145, 16_384),
    (262_145, 8192),
    (262_145, 4096),
    (262_145, 2048),
    (262_145, 1024),
    (524_289, 65_536),
    (524_289, 32_768),
    (524_289, 16_384),
    (524_289, 8192),
    (524_289, 4096),
    (524_289, 2048),
];

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
fn production_blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch =
        vec![Limb::MIN; bench_lopsided_mul_production_scratch_len(larger_len, smaller_len)];
    bencher.bench_local(|| {
        bench_lopsided_mul_production(
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
