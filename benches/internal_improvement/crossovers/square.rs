//! Where the squaring tower should hand over to the transform.
//!
//! `select_square_plan` gates its transform on `SSA_THRESHOLD`, a constant
//! measured entirely on *multiplication*. The two questions are not the same
//! one. A dedicated squaring tower is substantially cheaper than the general
//! product at the same width, because every tier squares its evaluated points
//! instead of multiplying pairs of them, while a transform saves comparatively
//! less on a square -- it still runs a forward transform, pointwise squares and
//! an inverse. Both of the transform's competitors therefore get cheaper, and
//! the width at which the transform wins should be *higher* for squaring than
//! for multiplication.
//!
//! Nothing had measured that, so this forces the transform against the
//! dispatcher across the crossover region, with GMP's dedicated `mpn_sqr` for
//! scale.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_sqr with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    state::SquareBenchScratch,
    transform::{bench_ssa_sqr_scratch_len, bench_ssa_sqr_with_scratch},
};

use crate::shared::operand;

const WIDTHS: [usize; 9] = [1800, 2000, 2200, 2400, 2500, 2600, 2800, 3000, 3072];

#[divan::bench(args = WIDTHS)]
fn dispatched(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut reusable = SquareBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(black_box(&mut destination), black_box(&value));
    });
}

#[divan::bench(args = WIDTHS)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut scratch = vec![Limb::MIN; bench_ssa_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_ssa_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = WIDTHS)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let count = size_t::try_from(len).expect("width fits a GMP size");
    bencher.bench_local(|| {
        // SAFETY: two independently allocated, disjoint vectors; the destination
        // holds exactly the 2n limbs mpn_sqr writes for an n-limb operand.
        unsafe {
            gmp::mpn_sqr(
                black_box(destination.as_mut_ptr().cast::<limb_t>()),
                black_box(value.as_ptr().cast::<limb_t>()),
                black_box(count),
            );
        }
        let _output = black_box(&destination);
    });
}
