//! Forced transform exponents at the RAM-sized rings the profile pins.
//!
//! `ssa/plan/tests.rs` pins the planner's exponent at four RAM-sized rings
//! to measured optima. Those pins are only meaningful while they match what the
//! hardware actually prefers, and the cost model reprices every nested geometry
//! when `SSA_BASE_MODULUS_BITS` moves, so a pin that changes has to be
//! re-measured rather than re-written.
//!
//! Operands here are millions of limbs, so this is deliberately excluded from
//! ordinary benchmark runs by its size rather than by a feature gate.

use core::hint::black_box;

use mp_anafis::tune_api::tier::{
    Limb,
    transform::{
        bench_ssa_mul_forced_plan, bench_ssa_mul_forced_plan_scratch_len,
        bench_ssa_mul_scratch_len, bench_ssa_mul_with_scratch,
    },
};

use crate::shared::operands_pair;

/// `(operand limbs, forced exponent)`; exponent zero lets the planner choose.
///
/// Every width carries the planner's own choice *and* the whole exponent window
/// either side of it, rather than the one or two neighbours a pin was written
/// against. A pin can only be validated against the alternatives it beat, and a
/// window of one neighbour cannot distinguish "the planner is right" from "the
/// planner is one short of an optimum that keeps moving with the ring".
const PROBES: [(usize, u32); 35] = [
    (262_144, 0),
    (262_144, 9),
    (262_144, 10),
    (262_144, 11),
    (262_144, 12),
    (262_144, 13),
    (1_048_576, 0),
    (1_048_576, 9),
    (1_048_576, 10),
    (1_048_576, 11),
    (1_048_576, 12),
    (1_048_576, 13),
    (2_097_152, 0),
    (2_097_152, 9),
    (2_097_152, 10),
    (2_097_152, 11),
    (2_097_152, 12),
    (2_097_152, 13),
    (4_194_304, 0),
    (4_194_304, 9),
    (4_194_304, 10),
    (4_194_304, 11),
    (4_194_304, 12),
    (4_194_304, 13),
    (8_388_608, 0),
    (8_388_608, 9),
    (8_388_608, 10),
    (8_388_608, 11),
    (8_388_608, 12),
    (8_388_608, 13),
    // The top of the `compare` ladder, and the widest ring the planner reaches.
    // Its exponent window sits two above the rings below it, so the neighbours
    // that matter here are not the ones that matter at 8M.
    (16_777_216, 0),
    (16_777_216, 11),
    (16_777_216, 12),
    (16_777_216, 13),
    (16_777_216, 14),
];

#[divan::bench(args = PROBES, sample_count = 3, sample_size = 1)]
fn transform(bencher: divan::Bencher<'_, '_>, probe: (usize, u32)) {
    let (len, exponent) = probe;
    let (larger, smaller, mut destination) = operands_pair(len, len);
    if exponent == 0 {
        let mut scratch = vec![Limb::MIN; bench_ssa_mul_scratch_len(len, len)];
        bencher.bench_local(|| {
            bench_ssa_mul_with_scratch(
                black_box(&mut destination),
                black_box(&larger),
                black_box(&smaller),
                black_box(&mut scratch),
            );
        });
        return;
    }
    let Some(scratch_len) = bench_ssa_mul_forced_plan_scratch_len(len, len, exponent) else {
        return;
    };
    let mut scratch = vec![Limb::MIN; scratch_len];
    bencher.bench_local(|| {
        bench_ssa_mul_forced_plan(
            black_box(&mut destination),
            black_box(&larger),
            black_box(&smaller),
            black_box(exponent),
            black_box(&mut scratch),
        );
    });
}
