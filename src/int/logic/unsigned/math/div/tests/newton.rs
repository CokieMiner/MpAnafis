//! Newton division is pinned against Algorithm D.
//!
//! Algorithm D is the exact reference: for every operand shape the reciprocal
//! path is used on, both halves of the result must match it limb for limb. The
//! last property crosses `NEWTON_RAPHSON_THRESHOLD` itself, where the dispatch
//! hands work to this module for the first time.

extern crate std;

use alloc::vec::Vec;
use core::cmp::Ordering;

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, Limb};

proptest! {
    #[test]
    fn prop_newton_terminates_for_normalized_denominators(
        n in prop_oneof![Just(1_usize), Just(2), Just(5), Just(10), Just(20), Just(40), Just(45), Just(60)],
        den_seed in any::<[u64; 60]>(),
        num_seed in any::<[u64; 120]>(),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed array length")
            .iter()
            .map(|limb| Limb::try_from(*limb).unwrap_or(0))
            .collect::<Vec<_>>();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed array length")
            .iter()
            .map(|limb| Limb::try_from(*limb).unwrap_or(0))
            .collect::<Vec<_>>();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut q_newton = InternalMpUint::zero();
        let mut r_newton = InternalMpUint::zero();
        Division::newton(&num, &den, &mut q_newton, &mut r_newton, &mut scratch);

        let mut q_expected = InternalMpUint::zero();
        let mut r_expected = InternalMpUint::zero();
        Division::algorithm_d(
            &num,
            &den,
            &mut q_expected,
            &mut r_expected,
            &mut scratch,
        );
        prop_assert_eq!(q_newton, q_expected, "quotient mismatch for n={}", n);
        prop_assert_eq!(r_newton, r_expected, "remainder mismatch for n={}", n);
    }

    #[test]
    fn prop_newton_reciprocal_and_div_match_algorithm_d(
        n in prop_oneof![Just(1_usize), Just(2), Just(5), Just(10), Just(20), Just(45), Just(60)],
        den_seed in any::<[u64; 60]>(),
        num_seed in any::<[u64; 120]>(),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed array length")
            .iter()
            .map(|limb| Limb::try_from(*limb).unwrap_or(0))
            .collect::<Vec<_>>();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed array length")
            .iter()
            .map(|limb| Limb::try_from(*limb).unwrap_or(0))
            .collect::<Vec<_>>();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut q_newton = InternalMpUint::zero();
        let mut r_newton = InternalMpUint::zero();
        Division::newton(&num, &den, &mut q_newton, &mut r_newton, &mut scratch);

        let mut q_expected = InternalMpUint::zero();
        let mut r_expected = InternalMpUint::zero();
        Division::algorithm_d(
            &num,
            &den,
            &mut q_expected,
            &mut r_expected,
            &mut scratch,
        );

        prop_assert_eq!(q_newton, q_expected, "Newton quotient mismatch for n={}", n);
        prop_assert_eq!(r_newton, r_expected, "Newton remainder mismatch for n={}", n);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn prop_newton_threshold_division_identity(
        n in prop_oneof![Just(640_usize), Just(641)],
        den_seed in proptest::collection::vec(any::<Limb>(), 641),
        num_seed in proptest::collection::vec(any::<Limb>(), 1_282),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed vector length")
            .to_vec();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed vector length")
            .to_vec();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        Division::newton(&num, &den, &mut quotient, &mut remainder, &mut scratch);
        prop_assert!(
            remainder.cmp(&den) == Ordering::Less,
            "remainder must be less than divisor for n={}",
            n,
        );
        let recombined = quotient.mul(&den).add(&remainder);
        prop_assert_eq!(recombined, num, "division identity failed for n={}", n);
    }
}
