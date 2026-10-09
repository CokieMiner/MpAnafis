//! Crossover-boundary coverage for the tuned GCD thresholds.
//!
//! Cases below, at, and above each crossover verify exact matrix action,
//! reduction invariants, and storage reuse.

use alloc::vec;

use crate::int::logic::unsigned::math::gcd::hgcd::hgcd_block;

#[cfg(feature = "std")]
use super::super::workspace::MAX_POOLED_HGCD_LIMBS;
#[cfg(feature = "std")]
use super::HGCD_CROSSOVER_THRESHOLD;
use super::{
    DivScratch, Gcd, HGCD_BLOCK_THRESHOLD, HgcdWorkspace, InternalMpUint,
    LEHMER_FUSED_UPDATE_MAX_LIMBS, LIMB_BITS, Limb, WIDE_LEHMER_THRESHOLD,
};

/// Automatic and forced update policies agree around the tuned width.
/// A shrinking pair selects its kernel from the current operand width.
#[test]
fn lehmer_fused_update_threshold_selects_the_kernel() {
    let boundary = LEHMER_FUSED_UPDATE_MAX_LIMBS;
    assert!(
        boundary >= 1,
        "the fused-update width must admit a one-limb operand"
    );
    for (width, expect_fused) in [
        (boundary.checked_sub(1).expect("positive fused width"), true),
        (boundary, true),
        (boundary.checked_add(1).expect("fused width fits"), false),
    ] {
        let (left, right) = ordered_pair(width, 23);
        // The exact transition (u, v) -> (v, u - v) applies at every width.
        // Testing the update directly also covers profiles whose fused cutoff
        // lies below the two-limb minimum required for quotient simulation.
        let (u0, v0, u1, v1, even) = (0, 1, 1, 1, false);
        let expected_u = right.clone();
        let expected_v = left.sub(&right);

        let mut automatic_u = left.clone();
        let mut automatic_v = right.clone();
        let mut automatic_next_u = InternalMpUint::zero();
        let mut automatic_next_v = InternalMpUint::zero();
        let automatic = Gcd::lehmer_update_dispatched(
            &mut automatic_u,
            &mut automatic_v,
            &mut automatic_next_u,
            &mut automatic_next_v,
            u0,
            v0,
            u1,
            v1,
            even,
            None,
        );

        let mut forced_u = left;
        let mut forced_v = right;
        let mut forced_next_u = InternalMpUint::zero();
        let mut forced_next_v = InternalMpUint::zero();
        let forced = Gcd::lehmer_update_dispatched(
            &mut forced_u,
            &mut forced_v,
            &mut forced_next_u,
            &mut forced_next_v,
            u0,
            v0,
            u1,
            v1,
            even,
            Some(expect_fused),
        );

        assert!(
            automatic,
            "the ordered pair must admit subtraction at {width} limbs"
        );
        assert_eq!((&automatic_u, &automatic_v), (&expected_u, &expected_v));
        assert_eq!(
            automatic, forced,
            "automatic and forced fused policy disagree at {width} limbs"
        );
        assert_eq!(
            (automatic_u, automatic_v),
            (forced_u, forced_v),
            "reduced pair differs between policies at {width} limbs"
        );
        assert_eq!(
            (automatic_next_u, automatic_next_v),
            (forced_next_u, forced_next_v),
            "scratch destinations differ between policies at {width} limbs"
        );
    }
}

/// The tuned wide-simulation width selects double-limb quotient simulation, and
/// whichever policy runs must describe a transition that preserves the gcd.
#[test]
fn wide_lehmer_threshold_selects_the_simulation_width() {
    // Double-limb extraction additionally requires three limbs, so the
    // observable boundary is the larger of that minimum and the tuned width.
    let boundary = WIDE_LEHMER_THRESHOLD.max(3);
    for (width, expect_wide) in [
        (boundary.checked_sub(1).expect("boundary above one"), false),
        (boundary, true),
        (boundary.checked_add(1).expect("boundary fits"), true),
    ] {
        let (left, right) = ordered_pair(width, 31);
        let automatic =
            Gcd::simulate_step::<false>(left.limbs(), right.limbs(), None, false, |_| {});
        let selected = Gcd::simulate_step::<false>(
            left.limbs(),
            right.limbs(),
            Some(expect_wide),
            false,
            |_| {},
        );
        assert_eq!(
            automatic, selected,
            "automatic simulation width differs from the selected policy at {width} limbs"
        );

        let expected = left.gcd(&right);
        let (u0, v0, u1, v1, even) = automatic;
        let mut u = left;
        let mut v = right;
        let mut next_u = InternalMpUint::zero();
        let mut next_v = InternalMpUint::zero();
        if Gcd::lehmer_update_dispatched(
            &mut u,
            &mut v,
            &mut next_u,
            &mut next_v,
            u0,
            v0,
            u1,
            v1,
            even,
            None,
        ) {
            assert_eq!(
                u.gcd(&v),
                expected,
                "an accepted transition changed the gcd at {width} limbs"
            );
        }
    }
}

#[test]
fn wide_lehmer_aligns_both_windows_at_every_normalization_shift() {
    for shift in 0..LIMB_BITS {
        let mut a_limbs = vec![Limb::MAX; 5];
        *a_limbs.last_mut().expect("five limbs") = Limb::MAX >> shift;
        let left = InternalMpUint::from_limbs(a_limbs);
        let mut b_limbs = vec![Limb::MAX; 5];
        *b_limbs.last_mut().expect("five limbs") = (Limb::MAX >> shift) >> 1 | 1;
        let right = InternalMpUint::from_limbs(b_limbs);
        let (head, divisor) = Gcd::extract_top_two_limbs(left.limbs(), right.limbs());
        let common_shift = 3_usize
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.checked_sub(shift))
            .expect("test shift fits");
        let mut expected_left = left.clone();
        let mut expected_right = right.clone();
        expected_left.shr_assign(common_shift);
        expected_right.shr_assign(common_shift);
        #[cfg(target_pointer_width = "64")]
        let (head_u128, divisor_u128) = (head, divisor);
        #[cfg(not(target_pointer_width = "64"))]
        let (head_u128, divisor_u128) = (u128::from(head), u128::from(divisor));
        assert_eq!(InternalMpUint::from_u128(head_u128), expected_left);
        assert_eq!(InternalMpUint::from_u128(divisor_u128), expected_right);

        let (u0, v0, u1, v1, even) =
            Gcd::simulate_step::<false>(left.limbs(), right.limbs(), Some(true), false, |_| {});
        let expected_gcd = left.gcd(&right);
        let mut u = left;
        let mut v = right;
        assert!(Gcd::lehmer_update_dispatched(
            &mut u,
            &mut v,
            &mut InternalMpUint::zero(),
            &mut InternalMpUint::zero(),
            u0,
            v0,
            u1,
            v1,
            even,
            None
        ));
        assert_eq!(u.gcd(&v), expected_gcd);
    }
}

/// The tuned block threshold bounds the recursive tier: below it the guard
/// rejects the block before touching the pair, and at or above it the reduction
/// must progress while preserving the gcd.
#[test]
fn hgcd_block_threshold_bounds_the_recursive_tier() {
    let boundary = HGCD_BLOCK_THRESHOLD;
    assert!(
        boundary >= 1,
        "the block threshold must admit a one-limb guard"
    );
    let mut scratch = DivScratch::default();
    let mut workspace = HgcdWorkspace::default();

    // Below the threshold the guard rejects the block, and rejection is a valid
    // state that must leave both operands exactly as they were.
    let below = boundary.checked_sub(1).expect("positive block threshold");
    let (below_left, below_right) = ordered_pair(below, 41);
    let mut rejected_u = below_left.clone();
    let mut rejected_v = below_right.clone();
    assert!(
        !hgcd_block::<false>(
            &mut rejected_u,
            &mut rejected_v,
            &mut scratch,
            &mut workspace,
            &mut 0
        ),
        "a block below the threshold must be rejected at {below} limbs"
    );
    assert_eq!(
        (rejected_u, rejected_v),
        (below_left, below_right),
        "a rejected block must leave the pair untouched"
    );

    // At and above it an ordered pair wider than the target must reduce.
    for width in [
        boundary,
        boundary.checked_add(1).expect("block threshold fits"),
    ] {
        let (left, right) = ordered_pair(width, 41);
        let expected = Gcd::compute_lehmer(&left, &right);
        let mut u = left;
        let mut v = right;
        assert!(
            hgcd_block::<false>(&mut u, &mut v, &mut scratch, &mut workspace, &mut 0),
            "a block at the threshold must make progress at {width} limbs"
        );
        assert!(
            u > v,
            "a completed block must leave its pair ordered at {width} limbs"
        );
        assert_eq!(
            u.gcd(&v),
            expected,
            "a completed block changed the gcd at {width} limbs"
        );
    }

    // `hgcd_step` delegates to the whole-block kernel while the `p = 2m/3`
    // partition leaves fewer than a block's worth of high limbs, and runs the
    // subquadratic step above that. Both sides must reduce and preserve the gcd.
    let delegates =
        |width: usize| width.wrapping_sub(width.wrapping_mul(2).wrapping_div(3)) < boundary;
    let mut step_width = boundary;
    while delegates(step_width) {
        step_width = step_width.wrapping_add(1);
    }
    let delegating = step_width.checked_sub(1).expect("boundary above one");
    assert!(
        delegating >= boundary && delegates(delegating) && !delegates(step_width),
        "the derived delegation boundary must straddle the partition rule"
    );
    for width in [delegating, step_width] {
        let (left, right) = ordered_pair(width, 43);
        let expected = Gcd::compute_lehmer(&left, &right);
        let mut u = left;
        let mut v = right;
        assert!(
            Gcd::hgcd_step::<false>(&mut u, &mut v, &mut scratch, &mut workspace, &mut 0),
            "an HGCD step must make progress at {width} limbs"
        );
        assert_eq!(
            u.gcd(&v),
            expected,
            "an HGCD step changed the gcd at {width} limbs"
        );
    }
}

/// Builds an ordered pair of exactly `width` significant limbs.
/// The recurrence is modulo the limb base. Odd limbs retain the requested
/// width, and ordering satisfies the Lehmer and HGCD input contracts.
fn ordered_pair(width: usize, seed: usize) -> (InternalMpUint, InternalMpUint) {
    let mut state = seed;
    let mut draw = |step: usize| {
        InternalMpUint::from_limbs(
            (0..width)
                .map(|_| {
                    state = state.wrapping_mul(step).wrapping_add(1);
                    state | 1
                })
                .collect(),
        )
    };
    let first = draw(40_503);
    let second = draw(34_283);
    if second > first {
        (second, first)
    } else {
        (first, second)
    }
}

#[cfg(feature = "std")]
#[test]
fn hgcd_callbacks_run_once_with_pooling_reentry_and_unwinding() {
    use core::{cell::Cell, panic::AssertUnwindSafe};
    use std::panic::catch_unwind;

    for mode in 0_u8..4 {
        let calls = Cell::new(0_u8);
        let payload = vec![mode];
        let operand_len = if mode == 1 {
            MAX_POOLED_HGCD_LIMBS
                .checked_add(1)
                .expect("pool bound fits")
        } else {
            0
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            HgcdWorkspace::with_thread_local(operand_len, |workspace| {
                calls.set(calls.get().checked_add(1).expect("at most two callbacks"));
                assert_eq!(payload.into_iter().sum::<u8>(), mode);
                workspace.locals.u = InternalMpUint::from_limb(71);
                if mode == 2 {
                    HgcdWorkspace::with_thread_local(0, |inner| {
                        calls.set(calls.get().checked_add(1).expect("one nested callback"));
                        assert!(inner.locals.u.is_zero(), "reentry uses separate storage");
                        inner.locals.u = InternalMpUint::from_limb(97);
                    });
                    assert_eq!(workspace.locals.u, InternalMpUint::from_limb(71));
                }
                assert_ne!(mode, 3, "callback unwind probe");
                11
            })
        }));
        assert_eq!(calls.get(), if mode == 2 { 2 } else { 1 });
        if mode == 3 {
            assert!(result.is_err(), "the callback panic must propagate");
        } else {
            assert_eq!(result.expect("nonpanicking callback"), 11);
        }
        HgcdWorkspace::with_thread_local(0, |workspace| {
            assert!(workspace.locals.u.is_zero(), "completed work clears locals");
        });
    }
}

/// The pooled HGCD workspace must be reachable from a thread-local destructor.
///
/// Thread-local destructors run in reverse order of first access, so a slot
/// touched before the first reduction is destroyed after the workspace slot and
/// observes an already-torn-down workspace. The reduction has to run on a
/// detached workspace there instead of panicking with `AccessError`, which
/// inside a destructor aborts the process.
#[cfg(feature = "std")]
#[test]
fn hgcd_workspace_survives_thread_local_teardown() {
    use core::cell::RefCell;
    /// Reduces a wide pair from its destructor, after the workspace slot is gone.
    struct LateReduction {
        left: InternalMpUint,
        right: InternalMpUint,
        expected: InternalMpUint,
    }

    impl Drop for LateReduction {
        fn drop(&mut self) {
            assert_eq!(
                self.left.gcd(&self.right),
                self.expected,
                "a reduction running after workspace teardown must stay exact"
            );
        }
    }

    std::thread_local! {
        static LATE: RefCell<Option<LateReduction>> = const { RefCell::new(None) };
    }

    // Register the client slot first, then force the workspace slot to register
    // its own destructor, so the client is destroyed last.
    let (left, right) = ordered_pair(HGCD_CROSSOVER_THRESHOLD, 97);
    let expected = left.gcd(&right);
    std::thread::spawn(move || {
        LATE.with(|slot| {
            *slot.borrow_mut() = Some(LateReduction {
                left,
                right,
                expected,
            });
        });
        // Touch the pooled workspace so its destructor is registered after the
        // client slot above. A second operand set keeps the probe independent
        // of the values the client destructor consumes.
        let (probe_left, probe_right) = ordered_pair(HGCD_CROSSOVER_THRESHOLD, 131);
        let expected_probe = probe_left.gcd(&probe_right);
        HgcdWorkspace::with_thread_local(HGCD_CROSSOVER_THRESHOLD, |workspace| {
            workspace.prepare(HGCD_CROSSOVER_THRESHOLD);
            assert!(!workspace.frames.is_empty());
        });
        assert!(!expected_probe.is_zero());
    })
    .join()
    .expect("thread teardown with a pooled reduction must not abort");
}

#[cfg(feature = "std")]
#[cfg_attr(
    miri,
    ignore = "The pool-limit fixture allocates more than 262144 limbs on 32- and 64-bit targets; bounded pool admission runs under Miri."
)]
#[test]
fn pooled_hgcd_releases_oversized_exact_multiple_storage() {
    std::thread::spawn(|| {
        let period = HGCD_CROSSOVER_THRESHOLD.max(3);
        let repetitions = MAX_POOLED_HGCD_LIMBS
            .checked_div(period)
            .and_then(|count| count.checked_add(2))
            .expect("the operand width fits")
            | 1;
        let shift = period.checked_mul(LIMB_BITS).expect("the period fits");
        let mut divisor = InternalMpUint::one();
        divisor.shl_assign(shift);
        divisor.increment();
        let mut numerator = InternalMpUint::one();
        numerator.shl_assign(
            shift
                .checked_mul(repetitions)
                .expect("the numerator width fits"),
        );
        numerator.increment();
        assert!(numerator.limbs().len() > MAX_POOLED_HGCD_LIMBS);

        // For odd k, x^k + 1 is divisible by x + 1. The initial remainder
        // completes this GCD before any recursive HGCD frame is prepared.
        assert_eq!(numerator.gcd(&divisor), divisor);
        drop(numerator);
        drop(divisor);
        HgcdWorkspace::with_thread_local(0, |workspace| {
            assert_eq!(workspace.max_prepared_len, 0);
            assert!(workspace.locals.rem.capacity() <= MAX_POOLED_HGCD_LIMBS);
            assert!(workspace.locals.scratch.u_norm.capacity() <= MAX_POOLED_HGCD_LIMBS);
        });
    })
    .join()
    .expect("oversized GCD must release its workspace");
}

#[cfg(feature = "std")]
#[test]
fn hgcd_pool_admission_preserves_reuse_across_its_boundary() {
    std::thread::spawn(|| {
        let capacity = 16;
        HgcdWorkspace::with_thread_local(capacity, |workspace| {
            workspace.locals.u.reserve(capacity);
        });
        for (width, pooled) in [
            (
                MAX_POOLED_HGCD_LIMBS
                    .checked_sub(1)
                    .expect("positive pool bound"),
                true,
            ),
            (MAX_POOLED_HGCD_LIMBS, true),
            (
                MAX_POOLED_HGCD_LIMBS
                    .checked_add(1)
                    .expect("pool bound fits"),
                false,
            ),
        ] {
            HgcdWorkspace::with_thread_local(width, |workspace| {
                assert_eq!(workspace.locals.u.capacity() >= capacity, pooled);
            });
        }
        HgcdWorkspace::with_thread_local(capacity, |workspace| {
            assert!(
                workspace.locals.u.capacity() >= capacity,
                "detached work must preserve the pooled buffer"
            );
        });
    })
    .join()
    .expect("pool admission must preserve reusable storage");
}
