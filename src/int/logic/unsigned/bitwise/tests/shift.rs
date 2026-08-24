//! Regression and property tests for unsigned shifts.

use alloc::vec;

use proptest::prelude::*;

use super::*;

#[test]
fn shl_word_shift_zeroes_low_limbs() {
    // A single bit in the top of limb 1. Shifting by two full limbs plus one
    // partial bit must land exactly on the boundary of limb 4 with every low
    // limb zero. Both shift amounts are limb-width independent: `LIMB_BITS`
    // is 16, 32, or 64, so `2 * LIMB_BITS + 1` never overflows and
    // `LIMB_BITS - 1` is always a valid shift amount.
    const SHIFT: usize = 2 * LIMB_BITS + 1;
    let hi = 1_usize << (LIMB_BITS - 1);
    let a = InternalMpUint::from_limbs(vec![0, hi]);
    let shifted = a.shl(SHIFT);
    let expected = InternalMpUint::from_limbs(vec![0, 0, 0, 0, 1]);
    assert_eq!(shifted, expected);
}

proptest! {
    #[test]
    fn shl_matches_shl_assign_reference(
        limbs_a in proptest::collection::vec(any::<Limb>(), 1..=8),
        shift in 1_usize..=LIMB_BITS.saturating_mul(5),
    ) {
        let a = InternalMpUint::from_limbs(limbs_a);
        let mut reference = a.clone();
        reference.shl_assign(shift);
        prop_assert_eq!(a.shl(shift), reference);

        let word_shift = shift.wrapping_div(LIMB_BITS);
        let result = a.shl(shift);
        if word_shift > 0 {
            let limbs = result.limbs();
            prop_assert!(limbs.len() >= word_shift);
            prop_assert!(limbs.iter().take(word_shift).all(|&l| l == 0));
        }
    }

    #[test]
    fn shr_matches_shr_assign_reference(
        limbs_a in proptest::collection::vec(any::<Limb>(), 1..=8),
        shift in 1_usize..=LIMB_BITS.saturating_mul(5),
    ) {
        let a = InternalMpUint::from_limbs(limbs_a);
        let mut reference = a.clone();
        reference.shr_assign(shift);
        prop_assert_eq!(a.shr(shift), reference);
    }
}
