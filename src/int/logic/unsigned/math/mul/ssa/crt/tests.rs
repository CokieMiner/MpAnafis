//! CRT reconstruction properties followed by guard and plan ownership cases.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Small explicit CRT fixtures bound all offsets and initialize guarded output windows before inspection"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::prelude::*;

use crate::int::{DoubleLimb, logic::unsigned::math::mul::Schoolbook};

use super::{CrtMulPlan, CrtSquarePlan, Limb, LimbOutput, SSA_BNM1_BASECASE_LIMBS, SsaCrt};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 32 }))]

    #[test]
    fn crt_merges_match_schoolbook_at_output_width_boundaries(
        left in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 4 } else { 32 }),
        right in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 4 } else { 32 }),
    ) {
        let n = left.len().max(right.len());
        let mut exact = vec![0; 2 * n];
        Schoolbook::mul(&mut exact, &left, &right);
        let mut xp = vec![0; n + 1];
        let mut xm = vec![0; n];
        // Reducing below B^(2n) subtracts the high half modulo B^n+1
        // and adds it modulo B^n-1. Schoolbook supplies the independent result.
        SsaCrt::stage_padded_difference(&mut xp, &exact);
        SsaCrt::fold_bnm1_product(&mut xm, &exact);
        check_exact_merges(&xp, &xm, &exact);
        // Both operands are below B^n; their product is below B^(2n)-1.
        check_recursive_merge(&xp, &xm, &exact);
    }
}

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The oracle widens native limbs to DoubleLimb and selects its radix-B digits exactly on 16/32/64-bit targets"
)]
fn crt_guard_borrow_and_redundant_zero_match_double_width_oracle() {
    let base = DoubleLimb::from(1_u8) << Limb::BITS;
    let mersenne = base - 1;
    for (low, guard, residue) in [
        (0, 0, Limb::MAX),             // Redundant quotient zero.
        (Limb::MAX, 0, 0),             // End-around subtraction borrow.
        (Limb::MAX - 1, 0, Limb::MAX), // Odd modular difference and low addition carry.
        (0, 1, 0),                     // Canonical Fermat guard.
        (Limb::MAX, 1, Limb::MAX - 1), // Semi-normal guard and escaping high carry.
    ] {
        let fermat = low as DoubleLimb + base * guard as DoubleLimb;
        let difference = (residue as DoubleLimb + mersenne - fermat % mersenne) % mersenne;
        let k = if difference.is_multiple_of(2) {
            difference >> 1
        } else {
            difference.midpoint(mersenne)
        };
        // k <= B-2 bounds k*(B+1) below B^2. A semi-normal Fermat
        // residue can cross B^2 once, which folds to one modulo B^2-1.
        let (integer, carry) = (k * (base + 1)).overflowing_add(fermat);
        let folded = (integer + DoubleLimb::from(carry)) % DoubleLimb::MAX;
        let xp = [low, guard];
        let xm = [residue];
        check_recursive_merge(&xp, &xm, &[folded as Limb, (folded >> Limb::BITS) as Limb]);
        if guard == 0 || low == 0 {
            assert!(!carry, "a canonical Fermat residue reconstructs below B^2");
            check_exact_merges(
                &xp,
                &xm,
                &[integer as Limb, (integer >> Limb::BITS) as Limb],
            );
        }
    }
}

#[test]
fn crt_plans_reject_invalid_ring_widths() {
    assert!(CrtMulPlan::new(0).is_none());
    assert!(CrtSquarePlan::new(0).is_none());
    let width = SSA_BNM1_BASECASE_LIMBS
        .checked_mul(2)
        .and_then(|doubled| doubled.checked_add(1))
        .expect("the configured CRT basecase leaves room for a recursive test");
    assert!(CrtMulPlan::new(width).is_none());
    assert!(CrtSquarePlan::new(width).is_none());
}

#[cfg(feature = "std")]
#[test]
fn crt_plan_storage_survives_cache_eviction() {
    let width = SSA_BNM1_BASECASE_LIMBS
        .checked_mul(2)
        .expect("a recursive CRT test width fits usize");
    let product = CrtMulPlan::new(width).expect("a doubled basecase admits a CRT split");
    let square = CrtSquarePlan::new(width).expect("a doubled basecase admits a CRT split");
    let product_shared = CrtMulPlan::new(width).expect("cached product geometry remains valid");
    let square_shared = CrtSquarePlan::new(width).expect("cached square geometry remains valid");
    assert_eq!(product.as_ptr(), product_shared.as_ptr());
    assert_eq!(square.as_ptr(), square_shared.as_ptr());

    // More distinct widths than retained slots evict both initial cache entries.
    // Live plans still own their levels while subsequent calls build replacements.
    for shift in 1..=5 {
        let next_width = width.checked_shl(shift).expect("test CRT widths fit usize");
        let _product = CrtMulPlan::new(next_width).expect("a power-of-two scaling admits CRT");
        let _square = CrtSquarePlan::new(next_width).expect("a power-of-two scaling admits CRT");
    }
    let product_rebuilt = CrtMulPlan::new(width).expect("evicted geometry remains valid");
    let square_rebuilt = CrtSquarePlan::new(width).expect("evicted geometry remains valid");
    assert_ne!(product.as_ptr(), product_rebuilt.as_ptr());
    assert_ne!(square.as_ptr(), square_rebuilt.as_ptr());
    assert_eq!(product.len(), product_rebuilt.len());
    assert_eq!(square.len(), square_rebuilt.len());
    assert_eq!(product.len(), product_shared.len());
    assert_eq!(square.len(), square_shared.len());
}

fn check_exact_merges(xp: &[Limb], xm: &[Limb], exact: &[Limb]) {
    let n = xm.len();
    let full_width = 2 * n;
    assert_eq!(xp.len(), n + 1);
    assert_eq!(exact.len(), full_width);
    let mut widths = vec![
        0,
        n - 1,
        n,
        n + 1,
        full_width - 1,
        full_width,
        full_width + 3,
    ];
    widths.sort_unstable();
    widths.dedup();
    for width in widths {
        let mut expected = exact.to_vec();
        expected.resize(width, 0);
        for in_place in [false, true] {
            if in_place && width < xp.len() {
                continue;
            }
            let mut output = vec![MaybeUninit::uninit(); width + 2];
            output[0] = MaybeUninit::new(37);
            output[width + 1] = MaybeUninit::new(37);
            let mut remainder = xm.to_vec();
            if in_place {
                for (destination, &digit) in output[1..=xp.len()].iter_mut().zip(xp) {
                    *destination = MaybeUninit::new(digit);
                }
                // SAFETY: the nonzero half-width and initialized canonical
                // Fermat prefix fit this disjoint writable output window.
                unsafe {
                    SsaCrt::merge_exact_product_in_place(&mut output[1..=width], &mut remainder);
                }
            } else {
                SsaCrt::merge_exact_product(&mut output[1..=width], xp, &mut remainder);
            }
            // SAFETY: either merge initialized the entire requested window,
            // including any zero suffix; both disjoint canaries started at 37.
            let initialized = unsafe { LimbOutput::assume_init(&output) };
            assert_eq!(
                &initialized[1..=width],
                expected,
                "width={width}, in_place={in_place}, xp={xp:?}, xm={xm:?}"
            );
            assert_eq!((initialized[0], initialized[width + 1]), (37, 37));
        }
    }
}

fn check_recursive_merge(xp: &[Limb], xm: &[Limb], expected: &[Limb]) {
    let width = 2 * xm.len();
    assert_eq!(xp.len(), xm.len() + 1);
    assert_eq!(expected.len(), width);
    let mut output = vec![MaybeUninit::uninit(); width + 2];
    output[0] = MaybeUninit::new(37);
    output[width + 1] = MaybeUninit::new(37);
    for (destination, &digit) in output[1..=xp.len()].iter_mut().zip(xp) {
        *destination = MaybeUninit::new(digit);
    }
    let mut remainder = xm.to_vec();
    // SAFETY: the exact two-half output contains its initialized Fermat
    // prefix and is disjoint from the complete nonempty Mersenne residue.
    unsafe {
        SsaCrt::merge_crt_halves_in_place(&mut output[1..=width], &mut remainder);
    }
    // SAFETY: the recursive merge initialized both complete output halves;
    // both excluded canaries were initialized before execution.
    let initialized = unsafe { LimbOutput::assume_init_mut(&mut output) };
    let actual = &mut initialized[1..=width];
    if actual.iter().all(|&limb| limb == Limb::MAX) {
        actual.fill(0);
    }
    assert_eq!(actual, expected, "xp={xp:?}, xm={xm:?}");
    assert_eq!((initialized[0], initialized[width + 1]), (37, 37));
}
