//! Shift forms, limb carries, exact scaling, and floor division.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
fn shifts_match_assign_forms_native_arithmetic_and_wide_power_of_two_identities() {
    let check = |limbs: &[Limb], shift: usize| {
        let value = InternalMpUint::from_limbs_slice(limbs);
        let left = value.shl(shift);
        let right = value.shr(shift);
        let mut assigned_left = value.clone();
        let mut assigned_right = value.clone();
        assigned_left.shl_assign(shift);
        assigned_right.shr_assign(shift);
        assert_eq!(left, assigned_left);
        assert_eq!(right, assigned_right);
        assert_eq!(left.shr(shift), value);
        let scale = InternalMpUint::power_of_two(shift);
        assert_eq!(left, value.mul(&scale));
        assert_eq!(right, value.div(&scale));
        if let Some(native) = value.to_u128() {
            let expected = u32::try_from(shift)
                .ok()
                .and_then(|amount| native.checked_shr(amount))
                .unwrap_or(0);
            assert_eq!(right.to_u128(), Some(expected));
        }
        let zero_words = shift.div_euclid(LIMB_BITS);
        assert!(
            left.limbs().iter().take(zero_words).all(|limb| *limb == 0),
            "whole-limb scaling initializes low zero words"
        );
        assert!(
            value.shr(usize::MAX).is_zero(),
            "an oversized right shift yields zero"
        );
    };
    for width in [0, 1, 3, 4, 5, 8] {
        for shift in [
            0,
            1,
            LIMB_BITS.checked_sub(1).expect("positive limb width"),
            LIMB_BITS,
            LIMB_BITS.checked_add(1).expect("partial shift"),
            LIMB_BITS.checked_mul(5).expect("wide shift"),
        ] {
            check(&vec![Limb::MAX; width], shift);
        }
    }
    let carry_shift = LIMB_BITS
        .checked_mul(2)
        .and_then(|bits| bits.checked_add(1))
        .expect("carry fixture");
    let source = InternalMpUint::from_limbs(vec![
        0,
        1 << LIMB_BITS.checked_sub(1).expect("positive limb width"),
    ]);
    assert_eq!(source.shl(carry_shift).limbs(), [0, 0, 0, 0, 1]);
    assert!(
        InternalMpUint::zero().shl(usize::MAX).is_zero(),
        "scaling zero needs no output allocation"
    );
    let strategy = (
        collection::vec(any::<Limb>(), 0..=8),
        0_usize..=LIMB_BITS.checked_mul(5).expect("bounded shift"),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, shift)| {
            check(&limbs, shift);
            Ok(())
        })
        .expect("unsigned shift property");
}
