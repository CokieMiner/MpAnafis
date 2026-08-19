//! Which transform geometry the planner should pick just above the crossover.
//!
//! The shape matrix leaves one unexplained cell: a 10000 by 7500 product runs
//! 1.12x behind the reference while both of its ratio neighbours are ahead of
//! it. Every one of those shapes selects the same tier, so the tier is not the
//! variable -- the transform *geometry* is. This forces each exponent in the
//! planner's search window on the same operands and reports what the planner
//! chose alongside them.
//!
//! An exponent of zero means the production planner picks. Exponents the ring
//! cannot support are skipped, and report as an empty timing rather than
//! panicking, because the valid window depends on the ring width and so differs
//! between the shapes compared here.

#![allow(
    unsafe_code,
    reason = "the benchmark calls GMP's raw mpn_mul with disjoint, exactly sized vectors"
)]

use core::hint::black_box;

use gmp_mpfr_sys::gmp::{self, limb_t, size_t};
use mp_anafis::tune_api::tier::{
    Tuner,
    transform::{SsaGeometryPolicy, SsaScratchPolicy, TransformExecutor},
};

use crate::shared::{gmp_pair_reference, operands_pair, validate_and_warm_product};

const PROBES: [(usize, usize, u32); 40] = [
    (10000, 7500, 0),
    (10000, 7500, 6),
    (10000, 7500, 7),
    (10000, 7500, 8),
    (10000, 7500, 9),
    (10000, 7500, 10),
    (10000, 7500, 11),
    (10000, 7500, 12),
    (10000, 7500, 13),
    (10000, 7500, 14),
    (10000, 8000, 0),
    (10000, 8000, 6),
    (10000, 8000, 7),
    (10000, 8000, 8),
    (10000, 8000, 9),
    (10000, 8000, 10),
    (10000, 8000, 11),
    (10000, 8000, 12),
    (10000, 8000, 13),
    (10000, 8000, 14),
    (10000, 6666, 0),
    (10000, 6666, 6),
    (10000, 6666, 7),
    (10000, 6666, 8),
    (10000, 6666, 9),
    (10000, 6666, 10),
    (10000, 6666, 11),
    (10000, 6666, 12),
    (10000, 6666, 13),
    (10000, 6666, 14),
    (10000, 10000, 0),
    (10000, 10000, 6),
    (10000, 10000, 7),
    (10000, 10000, 8),
    (10000, 10000, 9),
    (10000, 10000, 10),
    (10000, 10000, 11),
    (10000, 10000, 12),
    (10000, 10000, 13),
    (10000, 10000, 14),
];

#[divan::bench(args = PROBES)]
fn geometry(bencher: divan::Bencher<'_, '_>, probe: (usize, usize, u32)) {
    let (larger_len, smaller_len, exponent) = probe;
    let (larger, smaller, mut destination) = operands_pair(larger_len, smaller_len);
    let geometry = if exponent == 0 {
        SsaGeometryPolicy::Production
    } else {
        SsaGeometryPolicy::ForcedExponent(exponent)
    };
    let Some(mut runner) = Tuner::bench_ssa_multiplication(
        geometry,
        TransformExecutor::Sequential,
        SsaScratchPolicy::Reusable,
        &larger,
        &smaller,
    ) else {
        return;
    };
    let expected = gmp_pair_reference(&larger, &smaller);
    validate_and_warm_product(&expected, "SSA geometry product", |candidate| {
        runner.prepare(candidate).run();
    });
    let mut prepared = runner.prepare(&mut destination);
    bencher.bench_local(|| black_box(&mut prepared).run());
}

#[divan::bench(args = PROBES)]
fn gmp_reference(bencher: divan::Bencher<'_, '_>, probe: (usize, usize, u32)) {
    let (larger_len, smaller_len, exponent) = probe;
    if exponent != 0 {
        return;
    }
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
