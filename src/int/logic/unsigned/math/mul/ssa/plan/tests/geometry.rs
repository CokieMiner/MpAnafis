//! Ring partitions, primitive roots, sign separation, and scratch overflow.

use core::num::NonZeroUsize;

use alloc::vec;

use proptest::prelude::*;

use super::{super::super::SsaCrt, FftPlan, Geometry, LIMB_BITS};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 24 }))]

    #[test]
    fn selected_geometries_preserve_ring_and_workspace_invariants(
        exponent in 6_u32..if cfg!(miri) { 17 } else { usize::BITS },
        multiplier in prop::sample::select(vec![1_usize, 3, 5, 7, 9, 17, 31]),
    ) {
        if let Some(bits) = (1_usize << exponent).checked_mul(multiplier) { check_plan(bits); }
    }
}

#[cfg_attr(
    miri,
    ignore = "the full planner sweep covers large virtual geometries with recursive cost searches"
)]
#[test]
fn geometry_sweep_covers_power_of_two_and_irregular_moduli() {
    for exponent in 6_u32..30 {
        for multiplier in [1_usize, 3, 5, 7, 9, 17, 31] {
            if let Some(bits) = (1_usize << exponent).checked_mul(multiplier) {
                check_plan(bits);
            }
        }
    }
}

#[test]
fn aligned_candidates_keep_a_sign_bit_and_executor_sizes_preserve_workers() {
    let maximum = if cfg!(miri) { LIMB_BITS * 8 } else { 32768 };
    for bits in (LIMB_BITS..=maximum).step_by(LIMB_BITS) {
        for exponent in 1..bits.trailing_zeros() {
            for whole in [false, true] {
                for geometry in Geometry::for_exponent_candidates(exponent, bits, whole)
                    .into_iter()
                    .flatten()
                {
                    assert!(
                        2 * geometry.chunk_bits.get() + geometry.transform_log
                            < geometry.inner_bits
                    );
                }
            }
        }
    }
    let geometry = Geometry::for_exponent_candidates(8, 31744, false)
        .into_iter()
        .flatten()
        .next()
        .expect("divisible geometry");
    assert!(geometry.inner_bits > 256);
    for (workers, expected) in [(1, 1), (6, 6), (12, 12), (96, 64)] {
        assert_eq!(FftPlan::pointwise_parallelism_budget(64, workers), expected);
    }
    let plan = FftPlan::new(1 << 18);
    for (workers, slots) in [(1, 2), (6, 8), (12, 16)] {
        assert_eq!(plan.parallel_slots(workers), slots);
    }
    assert_eq!(SsaCrt::layout_len(usize::MAX, 1, 1, 0, 0), usize::MAX);
    assert_eq!(SsaCrt::sqr_layout_len(usize::MAX, 1, 1, 0, 0), usize::MAX);
    assert_eq!(
        SsaCrt::layout_len_concurrent(usize::MAX, 1, 1, 0, 0),
        usize::MAX
    );
    assert_eq!(
        SsaCrt::mul_mod_bnm1_scratch_len_for_parallelism(usize::MAX, 1),
        usize::MAX
    );
}

fn check_plan(bits: usize) {
    let plan = FftPlan::new(bits);
    if plan.mat_limbs.get() == usize::MAX {
        assert_eq!(
            plan.transform_mul_scratch_with_pointwise(1, NonZeroUsize::MIN),
            usize::MAX
        );
        return;
    }
    let logarithm = usize::try_from(plan.transform_len.trailing_zeros()).expect("logarithm fits");
    assert!(
        plan.chunk_bits
            .get()
            .checked_mul(2)
            .and_then(|n| n.checked_add(logarithm))
            .expect("coefficient bound fits")
            < plan.inner_bits
    );
    assert_eq!(
        plan.transform_len.checked_mul(plan.chunk_bits.get()),
        Some(bits)
    );
    assert!(plan.inner_bits.is_multiple_of(LIMB_BITS));
    let period = plan.inner_bits.checked_mul(2).expect("whole period fits");
    assert_eq!(plan.periods.whole.get(), period);
    assert_eq!(
        plan.periods.half.get(),
        period.checked_mul(2).expect("half period fits")
    );
    assert!(period.is_multiple_of(plan.transform_len));
    assert_eq!(
        plan.twist_step_half.checked_mul(plan.transform_len),
        Some(period)
    );
    assert!(plan.twist_step_half > 0);
}
