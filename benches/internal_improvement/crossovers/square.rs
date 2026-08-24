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

#![expect(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_sqr with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t};
use mp_anafis::tune_api::{
    SquaringAlgorithm, Tuner,
    tier::{Limb, state::SquaringBenchState},
};

use crate::shared::{operand, validated_gmp_count};

const WIDTHS: [usize; 9] = [1800, 2000, 2200, 2400, 2500, 2600, 2800, 3000, 3072];

#[divan::bench(args = WIDTHS)]
fn dispatched(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut reusable = SquaringBenchState::default();
    let mut prepared = reusable.prepare(&mut destination, &value);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}

#[divan::bench(args = WIDTHS)]
fn forced_transform(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let mut runner = Tuner::squaring(SquaringAlgorithm::SsaForced, len);
    let mut prepared = runner.prepare(&mut destination, &value);
    bencher.bench_local(|| {
        black_box(&mut prepared).run();
    });
}

#[divan::bench(args = WIDTHS)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, len: usize) {
    let value = operand(len, Limb::MAX.wrapping_sub(0x1234));
    let mut destination = vec![Limb::MIN; len.saturating_mul(2)];
    let count = validated_gmp_count(len);
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
