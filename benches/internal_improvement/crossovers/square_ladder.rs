//! The conventional squaring ladder either side of its Toom crossovers.
//!
//! This was built to chase a suspected misplaced crossover and found none. It is
//! kept because the negative result is worth as much as the search: three
//! squaring crossovers sit inside this band -- Karatsuba to Toom-3 at 192 limbs,
//! Toom-3 to Toom-4 at 256, and Toom-4 to Toom-6 at 320 -- and they are now
//! measured to be well placed rather than merely inherited.
//!
//! The suspicion came from a single run of `square_crossover` reporting 256
//! limbs at 1.04x against the reference while every neighbour sat far ahead.
//! Forcing each tier across the whole band does not reproduce it: 256 limbs
//! measures 0.72x, the band runs 0.60x to 0.84x throughout, and the dispatched
//! path is within 1% of the best forced tier at every width up to 384 and
//! strictly better than all of them from 448 up, because it recurses into
//! different tiers below the top level. The 1.04x reading was noise.
//!
//! Each tier is forced across the whole band rather than at its own crossover,
//! because the question is which tier *should* own each width, and a dispatcher
//! that already answers it cannot be asked.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_sqr with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Limb,
    algorithms::{
        bench_karatsuba_sqr_scratch_len, bench_karatsuba_sqr_with_scratch,
        bench_toom_cook_3_sqr_forced_scratch_len, bench_toom_cook_3_sqr_forced_with_scratch,
        bench_toom_cook_4_sqr_scratch_len, bench_toom_cook_4_sqr_with_scratch,
        bench_toom_cook_6_sqr_scratch_len, bench_toom_cook_6_sqr_with_scratch,
    },
    state::SquareBenchScratch,
};

use crate::shared::operand;

const WIDTHS: [usize; 12] = [128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 800];

fn setup(len: usize) -> (Vec<Limb>, Vec<Limb>) {
    (
        operand(len, Limb::MAX.wrapping_sub(0x1234)),
        vec![Limb::MIN; len.saturating_mul(2)],
    )
}

#[divan::bench(args = WIDTHS)]
fn dispatched(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
    let mut reusable = SquareBenchScratch::default();
    bencher.bench_local(|| {
        reusable.run(black_box(&mut destination), black_box(&value));
    });
}

#[divan::bench(args = WIDTHS)]
fn karatsuba(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
    let mut scratch = vec![Limb::MIN; bench_karatsuba_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_karatsuba_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = WIDTHS)]
fn toom3(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_3_sqr_forced_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_3_sqr_forced_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = WIDTHS)]
fn toom4(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_4_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_4_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = WIDTHS)]
fn toom6(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
    let mut scratch = vec![Limb::MIN; bench_toom_cook_6_sqr_scratch_len(len)];
    bencher.bench_local(|| {
        bench_toom_cook_6_sqr_with_scratch(
            black_box(&mut destination),
            black_box(&value),
            black_box(&mut scratch),
        );
    });
}

#[divan::bench(args = WIDTHS)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, len: usize) {
    let (value, mut destination) = setup(len);
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
