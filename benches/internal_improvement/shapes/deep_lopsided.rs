//! The one shape where neither top-level path is competitive.
//!
//! A 262145 by 2048 product measures 1.14x behind the reference dispatched, and
//! forcing the blocked path instead measures 1.13x to 1.21x, so unlike every
//! other gap closed so far this is not a routing mistake -- both candidates are
//! poor. Its neighbours are not: the same longer operand against 1024 limbs runs
//! 0.91x, and the same ratio at 131073 limbs is also ahead.
//!
//! This sweeps the shorter operand and the longer one independently around that
//! point, against GMP, with the blocked path forced at several block widths so
//! the block geometry is separated from the tier choice.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb, Tuner,
    state::MultiplicationBenchState,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{gmp_pair_reference, operands_pair, validate_and_warm_product};

const SHAPES: [(usize, usize); 10] = [
    (131_073, 2048),
    (262_145, 1024),
    (262_145, 1536),
    (262_145, 2048),
    (262_145, 3072),
    (262_145, 4096),
    (262_145, 8192),
    (524_289, 2048),
    (524_289, 4096),
    (1_048_577, 4096),
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

#[divan::bench(args = SHAPES)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut runner = Tuner::bench_ssa_multiplication(
        SsaGeometryPolicy::Forced,
        TransformExecutor::Sequential,
        SsaScratchPolicy::Reusable,
        &larger,
        &smaller,
    )
    .expect("forced SSA geometry is valid");
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "forced SSA deep-lopsided product", |probe| {
        runner.prepare(probe).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

/// Block-width sweep: the block is `MULTIPLIER` times the shorter operand.
///
/// `lopsided_block_len` currently caps the block just above the shorter operand
/// so that each block product is balanced enough for the widest conventional
/// split. That is the choice under test: a wider block makes each product
/// unbalanced but large enough to reach a transform instead.
macro_rules! blocked_at {
    ($name:ident, $num:expr, $den:expr) => {
        #[divan::bench(args = SHAPES)]
        fn $name(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
            let block = shape.1.saturating_mul($num).div_ceil($den).max(1);
            blocked(bencher, shape, block);
        }
    };
}

blocked_at!(block_1x, 1, 1);
blocked_at!(block_1_125x, 9, 8);
blocked_at!(block_2x, 2, 1);
blocked_at!(block_4x, 4, 1);
blocked_at!(block_8x, 8, 1);
blocked_at!(block_16x, 16, 1);
blocked_at!(block_32x, 32, 1);

fn blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize), block_len: usize) {
    let (larger_len, smaller_len) = shape;
    if block_len >= larger_len {
        return;
    }
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
