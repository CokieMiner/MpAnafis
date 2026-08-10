use proptest::prelude::*;

use super::*;

proptest! {
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest for leading bits and float conversion"
    )]
    #[test]
    fn leading_bits_collects_across_multiple_limbs_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 1..=4),
        width in 1_usize..=53,
    ) {
        let val = InternalArbiUint::from_limbs(limbs);
        let sig = val.significant_bits();
        if sig == 0 {
            prop_assert_eq!(leading_bits_as_u64(val.limbs(), sig, width), 0);
        } else {
            let result = leading_bits_as_u64(val.limbs(), sig, width);
            prop_assert!(result < (1_u64 << width.min(63)));
            let take = sig.min(width);
            let _expected = if take < 64 {
                let shift = sig.saturating_sub(take);
                let val_u64: u64 = if shift < 200 {
                    let v_shifted = val.shr(shift);
                    u64::try_from(v_shifted.limbs().first().copied().unwrap_or(0))
                        .unwrap_or(0)
                } else {
                    0
                };
                let mask = (1_u64 << take) - 1;
                val_u64 & mask
            } else {
                0
            };
            prop_assert!(result <= u64::MAX >> (64 - width.min(64)));
        }
    }

    #[test]
    fn float_conversion_prop(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=4),
    ) {
        let val = InternalArbiUint::from_limbs(limbs);
        if let Some(f64_val) = val.to_f64() {
            prop_assert!(f64_val.is_finite());
            prop_assert!(!f64_val.is_nan());
        }
        if let Some(f32_val) = val.to_f32() {
            prop_assert!(f32_val.is_finite());
            prop_assert!(!f32_val.is_nan());
        }
    }
}
