//! Exact odd division, signed shifts, and fused halving identities.

use alloc::vec::Vec;

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::InternalMpUint;

use super::super::{LIMB_BITS, Limb, SharedEval};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn odd_division_round_trips_modular_products_and_fused_subtractions(
        quotient in collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 48 }),
        odd_seed in any::<Limb>(), scalar in any::<Limb>(),
    ) {
        let width = quotient.len();
        let magnitude = InternalMpUint::from_limbs_slice(&quotient);
        let source_words: Vec<_> = quotient.iter().rev().copied().collect();
        let source = InternalMpUint::from_limbs_slice(&source_words);
        for divisor in [3, 15, 255, odd_seed | 1, Limb::MAX] {
            let inverse = SharedEval::invert_odd(divisor);
            prop_assert_eq!(divisor.wrapping_mul(inverse), 1);
            let product = magnitude.mul(&InternalMpUint::from_limb(divisor));
            let mut dividend: Vec<_> = product.limbs().iter().copied().take(width).collect();
            dividend.resize(width, 0);
            let mut specialized = dividend.clone();
            match divisor {
                3 => SharedEval::exact_div_radix_minus_one_in_place::<3>(&mut specialized),
                15 => SharedEval::exact_div_radix_minus_one_in_place::<15>(&mut specialized),
                255 => SharedEval::exact_div_radix_minus_one_in_place::<255>(&mut specialized),
                _ => SharedEval::exact_div_odd_in_place(&mut specialized, divisor, inverse),
            }
            prop_assert_eq!(specialized, quotient.clone());
            SharedEval::exact_div_odd_in_place(&mut dividend, divisor, inverse);
            prop_assert_eq!(dividend, quotient.clone());

            let combined = product.add(&source.mul(&InternalMpUint::from_limb(scalar)));
            let mut fused: Vec<_> = combined.limbs().iter().copied().take(width).collect();
            fused.resize(width, 0);
            SharedEval::exact_sub_mul_word_odd_in_place(&mut fused, &source_words, scalar, divisor);
            prop_assert_eq!(fused, quotient.clone());
        }
    }

    #[test]
    fn exact_shifts_and_halves_match_widening_and_modular_identities(
        words in collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 8 } else { 48 }),
        shift in 0_u32..Limb::BITS,
    ) {
        let mut divisible = words;
        if let Some(low) = divisible.first_mut() { *low &= Limb::MAX << shift; }
        let magnitude = InternalMpUint::from_limbs_slice(&divisible);
        let mut shifted = divisible.clone();
        SharedEval::exact_div_power_of_two_in_place(&mut shifted, shift);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&shifted), magnitude.shr(usize::try_from(shift).expect("shift fits")));
        let negative = divisible.last().is_some_and(|word| word >> (Limb::BITS - 1) != 0);
        if negative && shift != 0 {
            *shifted.last_mut().expect("negative value is nonempty") |= Limb::MAX << Limb::BITS.checked_sub(shift).expect("shift is below the limb width");
        }
        let mut signed = divisible.clone();
        SharedEval::exact_signed_div_power_of_two_in_place(&mut signed, shift);
        prop_assert_eq!(signed, shifted);

        if let Some(low) = divisible.first_mut() { *low &= !3; }
        let even = InternalMpUint::from_limbs_slice(&divisible);
        let mut half = divisible.clone();
        SharedEval::exact_div2_in_place(&mut half);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&half), even.shr(1));
        let mut quarter = divisible.clone();
        SharedEval::exact_div4_in_place(&mut quarter);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&quarter), even.shr(2));

        let mut widening = divisible.clone();
        SharedEval::exact_half_sum_in_place(&mut widening, &divisible);
        prop_assert_eq!(widening, divisible.clone());
        let mut modular = divisible.clone();
        SharedEval::exact_half_modular_sum_in_place(&mut modular, &divisible);
        let width_bits = divisible.len().checked_mul(LIMB_BITS).expect("width fits");
        let modulus = InternalMpUint::one().shl(width_bits);
        let expected = even.add(&even).div_rem(&modulus).1.shr(1);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&modular), expected);
        SharedEval::exact_half_reverse_difference_in_place(&mut half, &divisible);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&half), even.shr(2));
    }
}
