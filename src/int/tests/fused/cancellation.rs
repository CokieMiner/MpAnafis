//! Exact cancellation preserves magnitudes and validates only final bounded values.

extern crate std;

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::vec;

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{
    MpInt, MpUint,
    int::tests::support::{nz, uint_from_words},
};

proptest! {
    #[test]
    fn signed_subtraction_preserves_normalized_upper_words(limb_count in 2_usize..=12, delta in 1_usize..=1024) {
        // B^limb_count - delta has a short negated residue and nonzero upper words.
        let mut words = vec![usize::MAX; limb_count];
        *words.first_mut().expect("limb count is positive") = usize::MAX - (delta - 1);
        let magnitude = MpInt::from(uint_from_words(&words));
        let zero = MpInt::zero();
        let negative = -&magnitude;
        prop_assert_eq!(&(&zero - &magnitude), &negative);
        prop_assert_eq!(&(&zero + &negative), &negative);
        let mut difference = MpInt::zero();
        difference.assign_sub(&zero, &magnitude);
        prop_assert_eq!(&difference, &negative);
        let mut sum = MpInt::zero();
        sum.assign_add(&zero, &negative);
        prop_assert_eq!(&sum, &negative);
        let mut assigned_difference = MpInt::zero();
        assigned_difference -= &magnitude;
        prop_assert_eq!(&assigned_difference, &negative);
        let mut assigned_sum = MpInt::zero();
        assigned_sum += &negative;
        prop_assert_eq!(assigned_sum, negative);
    }

    #[test]
    fn fused_multiply_add_validates_the_exact_final_result(bits in 3_usize..=64, selection in any::<u64>()) {
        let maximum = (1_i128 << (bits - 1)) - 1;
        let lower = (maximum >> 1) + 1;
        let value = lower + i128::from(selection) % (maximum - lower + 1);
        let factor = MpInt::with_precision_checked(value, nz(bits)).expect("selected factor fits");
        let multiplier = MpInt::with_precision_checked(2_i8, nz(bits)).expect("two fits");
        let addend = MpInt::with_precision_checked(-value, nz(bits)).expect("negative factor fits");
        prop_assert!(catch_unwind(AssertUnwindSafe(|| &factor * &multiplier)).is_err());
        let fused = factor.mul_add(&multiplier, &addend);
        prop_assert_eq!(&fused, &factor);
        prop_assert_eq!(fused.precision(), factor.precision());
    }

    #[test]
    fn power_of_two_scaling_matches_shift_operators(
        unsigned in crate::int::tests::strategies::uint(8),
        signed in crate::int::tests::strategies::int(8), shift in 0_usize..=128,
    ) {
        prop_assert_eq!(unsigned.mul_2exp(shift), &unsigned << shift);
        prop_assert_eq!(unsigned.div_2exp(shift), &unsigned >> shift);
        prop_assert_eq!(signed.mul_2exp(shift), &signed << shift);
        prop_assert_eq!(signed.div_2exp(shift), &signed >> shift);
        let zero = MpUint::zero();
        prop_assert_eq!(&zero.mul_2exp(usize::MAX), &zero);
        prop_assert_eq!(zero.div_2exp(usize::MAX), zero);
    }
}
