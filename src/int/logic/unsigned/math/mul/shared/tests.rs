//! Properties for multiplication-tier fixed-width helpers.

use proptest::prelude::*;

use super::{Limb, SharedEval};

fn multiply_by_three_mod_width(value: &mut [Limb]) {
    let mut carry = 0;
    for limb in value {
        let original = *limb;
        let (twice, overflow_twice) = original.overflowing_add(original);
        let (thrice, overflow_thrice) = twice.overflowing_add(original);
        let (result, overflow_carry) = thrice.overflowing_add(carry);
        *limb = result;
        carry = Limb::from(overflow_twice)
            .wrapping_add(Limb::from(overflow_thrice))
            .wrapping_add(Limb::from(overflow_carry));
    }
}

proptest! {
    #[test]
    fn prop_exact_div3_round_trips_fixed_width(
        quotient in proptest::collection::vec(any::<Limb>(), 0..=48),
    ) {
        let mut dividend = quotient.clone();
        multiply_by_three_mod_width(&mut dividend);
        SharedEval::exact_div_radix_minus_one_in_place::<3>(&mut dividend);
        prop_assert_eq!(dividend, quotient);
    }
}
