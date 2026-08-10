//! Where the transform stops being worth padding the shorter operand.
//!
//! This sweep ran in two passes, and the shapes below are the second one. The
//! first swept ratio against width, six widths from 4091 to 65537 limbs by
//! ratios from six to forty to one, to test whether the flat eight-to-one bound
//! that `transform_padding_is_affordable` used to apply should instead rise with
//! the width. It should not: sorted by the *shorter* operand those 58 shapes
//! separated with no crossings and the longer operand predicted nothing, which
//! is why the predicate now asks about the shorter operand alone.
//!
//! What remained was where on that axis the boundary sits, so these shapes hold
//! the shorter operand near it — 900 to 1280 limbs — across five widths, and let
//! the width vary underneath. The answer is that the boundary is soft: the two
//! paths stay within about 10% of each other across the whole band and the
//! winner is not monotone in the width. Anything read from a single run of these
//! shapes at better than 10% resolution is noise.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use arbi_anafis::tune_api::tier::{
    Limb,
    state::{bench_lopsided_mul_scratch_len, bench_lopsided_mul_with_scratch},
    transform::{bench_ssa_mul_scratch_len, bench_ssa_mul_with_scratch},
};
use gmp_mpfr_sys::gmp::{self, limb_t, size_t};

use crate::shared::operands_pair;

const SHAPES: [(usize, usize); 25] = [
    (4091, 900),
    (4091, 1022),
    (4091, 1100),
    (4091, 1170),
    (4091, 1280),
    (6000, 900),
    (6000, 1022),
    (6000, 1100),
    (6000, 1170),
    (6000, 1280),
    (10000, 900),
    (10000, 1022),
    (10000, 1100),
    (10000, 1170),
    (10000, 1280),
    (16385, 900),
    (16385, 1022),
    (16385, 1100),
    (16385, 1170),
    (16385, 1280),
    (32769, 900),
    (32769, 1022),
    (32769, 1100),
    (32769, 1170),
    (32769, 1280),
];

/// The transform forced past the ratio at which the dispatcher denies it.
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

/// The blocked path the dispatcher currently selects for all of these.
#[divan::bench(args = SHAPES)]
fn forced_blocked(bencher: divan::Bencher<'_, '_>, shape: (usize, usize)) {
    let (larger_len, smaller_len) = shape;
    let block_len = smaller_len.saturating_add(smaller_len.div_ceil(8));
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let mut scratch =
        vec![Limb::MIN; bench_lopsided_mul_scratch_len(larger_len, smaller_len, block_len)];
    bencher.bench_local(|| {
        bench_lopsided_mul_with_scratch(
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
