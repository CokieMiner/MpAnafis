//! Regression tests for the SSA transform-geometry planner.
//!
//! The measured optima come from `algorithms::ssa_large_geometry` in
//! `benches/internal_improvement`, which times every forced exponent against
//! the planner's own choice. A geometry regression here costs a factor of two
//! or more end to end, so the selections are pinned.

#[cfg(target_arch = "x86_64")]
use super::{FftPlan, LIMB_BITS, SSA_BASE_MODULUS_BITS, SsaPlan};

use super::super::SsaCrt;

#[test]
fn crt_layout_rejects_unrepresentable_scratch_sizes() {
    assert_eq!(SsaCrt::layout_len(usize::MAX, 1), usize::MAX);
    assert_eq!(SsaCrt::sqr_layout_len(usize::MAX, 1), usize::MAX);
    assert_eq!(SsaCrt::mul_mod_bnm1_scratch_len(usize::MAX), usize::MAX);
}

/// Ring widths reached by the top-level Fermat product for power-of-two operand
/// widths, paired with the transform exponent the planner selects.
///
/// **Architecture Restriction:** The optima here and below are pinned to the
/// constants of the `x86_64` tuning profile (like `SSA_BASE_MODULUS_BITS` and
/// `SSA_NESTED_COST_PENALTY_16THS`). Other architectures use different profile
/// defaults, so the cost model will select different exponents there. These
/// exact-match tests are therefore restricted to `x86_64`.
///
/// Re-measured after the half-bit pre-twist opened the `sqrt(2)` geometries.
/// The selection is the measured optimum at every width except `1 << 24`, where
/// exponent 11 timed 50.74 ms against the selected 12 at 51.24 ms — a 1.0% model
/// error that is pinned deliberately rather than tuned away, because fitting the
/// cost model to a single width is how it stops generalising. Exponent 8 at
/// `1 << 19` is carried over: it is outside the sweep's range.
const MEASURED_OPTIMA: [(usize, usize); 7] = [
    (1 << 18, 7),  // 4,096-limb operands
    (1 << 19, 8),  // 8,192-limb operands
    (1 << 20, 9),  // 16,384-limb operands
    (1 << 21, 10), // 32,768-limb operands
    (1 << 22, 10), // 65,536-limb operands
    (1 << 23, 11), // 131,072-limb operands
    (1 << 24, 12), // 262,144-limb operands
];

/// RAM-sized equal-width cells measured on the x86-64 profile.
///
/// Re-measured at `1 << 26` after the nested exponent search moved onto
/// `log2/2` and `SSA_NESTED_COST_PENALTY_16THS` began charging for nesting.
/// Forcing the whole window on identical operands in one process timed
/// exponent 11 at 517.9 ms against the previously selected 10 at 533.3 ms,
/// with 9, 12 and 13 at 551.2, 552.0 and 529.8 ms.
///
/// Re-measured before that after the nested inner ring stopped being rounded up
/// to the next power of two. That rounding had made every nesting geometry cost up to twice
/// what it should, so the cost model preferred geometries whose pointwise stage
/// stayed in the tower and landed two to three exponents above the optimum. The
/// selections below are the model's own, and at 2M, 4M, and 8M limbs they
/// measured at or below every forced exponent in a five-wide sweep.
#[cfg(target_arch = "x86_64")]
const X86_RAM_OPTIMA: [(usize, usize); 4] = [
    (1 << 26, 11), // 1,048,576-limb operands
    (1 << 27, 12), // 2,097,152-limb operands
    (1 << 28, 12), // 4,194,304-limb operands
    (1 << 29, 13), // 8,388,608-limb operands
];

/// Ring widths the `mul_mod_bnm1` recursion reaches below the pinned range.
/// Their geometry is not pinned to a measured optimum, but it must stay valid
/// and must keep the pointwise stage in the tower.
#[cfg(target_arch = "x86_64")]
const RECURSION_RINGS: [usize; 3] = [1 << 15, 1 << 16, 1 << 17];

#[cfg(target_arch = "x86_64")]
#[test]
fn planner_matches_measured_optima() {
    for (modulus_bits, expected) in MEASURED_OPTIMA {
        let chosen = FftPlan::new(modulus_bits).transform_log;
        assert_eq!(
            chosen, expected,
            "planner chose exponent {chosen} for a {modulus_bits}-bit ring, \
             but exponent {expected} measured fastest"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_profile_matches_measured_ram_optima() {
    for (modulus_bits, expected) in X86_RAM_OPTIMA {
        let chosen = FftPlan::new(modulus_bits).transform_log;
        assert_eq!(
            chosen, expected,
            "x86 profile chose exponent {chosen} for a {modulus_bits}-bit RAM-sized ring"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
fn chosen_geometry_keeps_pointwise_in_the_basecase() {
    // Every ring the benchmarks cover admits a geometry whose inner ring stays
    // inside the multiplication tower. Falling back to a nested transform there
    // was the original planner's dominant failure mode.
    let pinned = MEASURED_OPTIMA.into_iter().map(|(bits, _)| bits);
    for modulus_bits in pinned.chain(RECURSION_RINGS) {
        let plan = FftPlan::new(modulus_bits);
        assert!(
            plan.inner_bits <= SSA_BASE_MODULUS_BITS,
            "planner selected a nested inner ring of {} bits for a {modulus_bits}-bit ring",
            plan.inner_bits
        );
    }
}

#[test]
// Miri is incredibly slow for exhaustive safe-math tests, and QEMU user-mode
// on 32-bit architectures frequently crashes with false stack overflows due to
// buggy thread stack mapping. The 64-bit native CI proves the planner logic.
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
#[cfg_attr(
    not(target_pointer_width = "64"),
    ignore = "QEMU user-mode on 32-bit architectures frequently crashes with false stack overflows; the 64-bit native CI proves the planner logic"
)]
fn every_selected_plan_is_a_valid_geometry() {
    // Sweep both power-of-two ring widths and the odd-multiplier widths the
    // CRT half-width selector produces, so the planner is exercised on the
    // non-power-of-two moduli the top level now hands it.
    for exponent in 6_u32..30 {
        let base = 1_usize << exponent;
        for multiplier in [1_usize, 3, 5, 7, 9, 17, 31] {
            let Some(modulus_bits) = base.checked_mul(multiplier) else {
                continue;
            };
            assert_plan_is_consistent(modulus_bits);
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
fn forced_geometries_agree_with_the_selected_one() {
    // `try_forced` and `new` must build identical plans from the same exponent.
    // The forced path is what the benchmarks and the forced scratch-length
    // calculation use, so a divergence there mis-sizes buffers.
    for (modulus_bits, _) in MEASURED_OPTIMA {
        let selected = FftPlan::new(modulus_bits);
        let exponent = u32::try_from(selected.transform_log).expect("exponent fits u32");
        let forced = FftPlan::try_forced(modulus_bits, exponent)
            .expect("the selected exponent is by construction a valid geometry");
        assert_eq!(forced.transform_len, selected.transform_len);
        assert_eq!(forced.inner_bits, selected.inner_bits);
        assert_eq!(forced.twist_step_half, selected.twist_step_half);
        assert_eq!(
            forced.transform_mul_scratch(),
            selected.transform_mul_scratch(),
            "forced and selected plans disagree on scratch for {modulus_bits} bits"
        );
    }
}

#[test]
fn planner_rejects_sub_half_efficiency_geometry() {
    let modulus_bits = 1 << 20;

    assert!(
        SsaPlan::price_geometry(12, modulus_bits, 0, false).is_none(),
        "a K=4096 split uses less than half of its aligned coefficient ring"
    );
}

#[test]
fn search_centre_is_stable_across_near_equal_widths() {
    // The centre must stay meaningful for non-power-of-two rings; anchoring it
    // to `trailing_zeros` collapsed once the CRT half-width stopped being a
    // power of two.
    let power_of_two = SsaPlan::search_centre(1 << 24);
    let odd_multiple = SsaPlan::search_centre(3 << 23);
    assert_eq!(
        power_of_two, odd_multiple,
        "rings of near-equal width must start the search at near-equal exponents"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
fn search_centre_keeps_measured_optima_inside_the_search_window() {
    // The centre exists to place the search window over the optimum. Pinning
    // the centre's own value would pin an implementation; this pins the
    // property that makes it correct, at every width where an optimum was
    // measured.
    for (modulus_bits, measured) in MEASURED_OPTIMA {
        assert_window_reaches(modulus_bits, measured);
    }
    #[cfg(target_arch = "x86_64")]
    for (modulus_bits, measured) in X86_RAM_OPTIMA {
        assert_window_reaches(modulus_bits, measured);
    }
}

fn assert_window_reaches(modulus_bits: usize, measured: usize) {
    let centre = SsaPlan::search_centre(modulus_bits);
    let optimum = u32::try_from(measured).expect("a transform exponent fits u32");
    assert!(
        centre.abs_diff(optimum) <= 4,
        "the search window around centre {centre} does not reach the measured \
         optimum {optimum} for a {modulus_bits}-bit ring"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
fn search_centre_does_not_drift_from_the_optimum_as_rings_grow() {
    // The defect this centre replaced was not a wrong value at any measured
    // width — it was a deviation that grew without bound, so the window stopped
    // containing the optimum somewhere above the widest cell anyone had run.
    // The classical split puts the optimum near `log2(bits) / 2`, so the centre
    // must stay a bounded distance from it at every ring width a `usize` can
    // describe, including the ones no benchmark reaches.
    for log2_bits in 6_u32..usize::BITS {
        let modulus_bits = 1_usize << log2_bits;
        let centre = SsaPlan::search_centre(modulus_bits);
        let classical = log2_bits.div_euclid(2);
        assert!(
            centre.abs_diff(classical) <= 1,
            "centre {centre} drifted from the classical split exponent {classical} \
             for a 2^{log2_bits}-bit ring"
        );
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri is far too slow for exhaustive safe-math sweeps; the native CI covers this behavior"
)]
fn one_centre_serves_both_search_levels() {
    // `nested_ring_bits` rounds an inner ring up to a multiple of the transform
    // length the *nested* planner may pick, so it must estimate that length from
    // the centre the nested search actually uses. There is now exactly one such
    // centre, which is what makes that guarantee hold by construction rather
    // than by two definitions agreeing.
    //
    // The nested centre used to be anchored to the basecase width instead, at
    // `log2(bits) - log2(SSA_BASE_MODULUS_BITS) + 2`. That deviates from the
    // classical optimum without bound, so a wide enough inner ring was priced
    // with a window that could not reach its own optimum. Decoupling the two
    // centres again would reintroduce that, so this pins the nested window onto
    // the classical split at every width the recursion can reach.
    for log2_bits in 16_u32..30 {
        let modulus_bits = 1_usize << log2_bits;
        let centre = SsaPlan::search_centre(modulus_bits);
        let classical = log2_bits.div_euclid(2);
        assert_eq!(
            centre, classical,
            "the nested centre for a 2^{log2_bits}-bit ring left the classical split"
        );
    }
}

#[test]
fn basecase_cost_is_monotone_and_subquadratic() {
    let mut previous = 0;
    for limbs in 1_usize..=512 {
        let cost = SsaPlan::basecase_product_cost(limbs);
        assert!(
            cost >= previous,
            "basecase cost fell from {previous} to {cost} at {limbs} limbs"
        );
        previous = cost;
    }
    // A quadratic model over-prices wide coefficients and pushes the planner
    // toward transforms that are too long; check the model really is below it.
    let wide = SsaPlan::basecase_product_cost(256);
    assert!(
        wide < 256_usize.saturating_mul(256),
        "basecase model is not subquadratic at 256 limbs: {wide}"
    );
}

/// Asserts the invariants `orchestrate`, `split`, and `reconstruct` rely on:
/// the transform partitions the ring exactly, the inner ring is a whole number
/// of limbs, and it carries a primitive root of the transform's order.
fn assert_plan_is_consistent(modulus_bits: usize) {
    let plan = FftPlan::new(modulus_bits);

    assert_eq!(
        plan.transform_len
            .checked_mul(plan.chunk_bits)
            .expect("chunk product fits"),
        modulus_bits,
        "geometry does not partition the {modulus_bits}-bit ring exactly"
    );
    assert!(
        plan.inner_bits.is_multiple_of(LIMB_BITS),
        "inner ring of {} bits is not a whole number of limbs",
        plan.inner_bits
    );
    // The transform root is 2^omega_shift and 2 has order 2 * inner_bits, so
    // the ring needs `transform_len | 2 * inner_bits` — half of what a
    // whole-bit pre-twist would demand, because the pre-twist may carry a
    // `sqrt(2)` factor.
    let doubled = plan
        .inner_bits
        .checked_mul(2)
        .expect("twice the ring width fits");
    assert!(
        doubled.is_multiple_of(plan.transform_len),
        "inner ring of {} bits has no exact {}-th root of unity",
        plan.inner_bits,
        plan.transform_len
    );
    assert_eq!(
        plan.twist_step_half
            .checked_mul(plan.transform_len)
            .expect("root period fits"),
        doubled,
        "half-bit twist step is not exact for a {modulus_bits}-bit ring"
    );
    assert!(
        plan.twist_step_half > 0,
        "root step collapsed to zero for a {modulus_bits}-bit ring"
    );
}
