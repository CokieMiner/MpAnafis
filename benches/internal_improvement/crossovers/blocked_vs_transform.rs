//! Whether the transform should pre-empt blocking at four-to-three.
//!
//! `prefers_blocked_product` fires from four-to-three upward, but the transform
//! tier is offered above it, so at exactly that ratio the blocked path is never
//! reached. The shape matrix leaves 10000 by 7500 at 1.12x behind the reference
//! with both ratio neighbours ahead of it, and a forced sweep of every transform
//! geometry in the planner's window put the best one at 1.10x -- so the geometry
//! is not the variable and the tier itself is the open question.
//!
//! Each shape runs the production dispatcher (which selects a transform at all
//! of these widths), the blocked path forced on the same operands, and GMP.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{Limb, Tuner, state::MultiplicationBenchState};

use crate::shared::operands_pair;

const SHAPES: [(usize, usize); 5] = [
    (6000, 4500),
    (10000, 7500),
    (13000, 9750),
    (16385, 12288),
    (32769, 24576),
];

#[divan::bench(args = SHAPES)]
fn dispatched(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut reusable = MultiplicationBenchState::default();
    bencher.bench_local(|| {
        reusable.run(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
        );
    });
}

/// The blocked path with the shorter operand's own width as the block.
#[divan::bench(args = SHAPES)]
fn forced_blocked_square(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    blocked(bencher, shape, shape.1);
}

/// The blocked path with the widest block a Toom-8 split still accepts, which is
/// production's other candidate width.
#[divan::bench(args = SHAPES)]
fn forced_blocked_toom8_half(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let block = shape.1.saturating_add(shape.1.div_ceil(8));
    blocked(bencher, shape, block);
}

fn blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize), block_len: usize) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
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
