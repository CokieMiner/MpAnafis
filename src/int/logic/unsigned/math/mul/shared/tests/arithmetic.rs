//! Scalar products, borrow chains, shifted reconstruction, and paired points.

#![expect(
    unsafe_code,
    reason = "The test reserves the complete shifted sum and compares its initialized output to arbitrary-precision arithmetic"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::InternalMpUint;

use super::super::{LIMB_BITS, Limb, SharedEval};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn fixed_width_arithmetic_matches_arbitrary_precision(
        inputs in collection::vec(collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 24 }), 4),
        scalar in any::<Limb>(), shift in 0_usize..=4,
    ) {
        let width = inputs.iter().map(Vec::len).max().expect("four inputs").checked_add(2).expect("guards fit");
        let values: Vec<_> = inputs.iter().map(|words| InternalMpUint::from_limbs_slice(words)).collect();
        let [original, first, second, third] = <&[InternalMpUint; 4]>::try_from(values.as_slice()).expect("four generated inputs");
        let modulus = InternalMpUint::one().shl(width.checked_mul(LIMB_BITS).expect("bit width fits"));
        let mut padded = inputs.first().expect("original input").clone();
        padded.resize(width, 0);
        let mut source = inputs.get(1).expect("first source").clone();
        source.resize(width, 0);
        let product = first.mul(&InternalMpUint::from_limb(scalar));

        let mut added = padded.clone();
        SharedEval::add_mul_word_in_place(&mut added, &source, scalar);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&added), original.add(&product));
        SharedEval::sub_mul_word_in_place(&mut added, &source, scalar);
        prop_assert_eq!(added, padded.clone());

        for count in 1..=3 {
            let mut difference = padded.clone();
            let total = match count {
                1 => { SharedEval::sub_full_slices_in_place(&mut difference, first.limbs()); first.clone() }
                2 => { SharedEval::sub_two_full_slices_in_place(&mut difference, first.limbs(), second.limbs()); first.add(second) }
                _ => { SharedEval::sub_three_full_slices_in_place(&mut difference, first.limbs(), second.limbs(), third.limbs()); first.add(second).add(third) }
            };
            let residue = total.div_rem(&modulus).1;
            let expected = original.add(&modulus).sub(&residue).div_rem(&modulus).1;
            prop_assert_eq!(InternalMpUint::from_limbs_slice(&difference), expected);
        }

        let mut sum = padded.clone();
        let mut difference = source.clone();
        let negative = SharedEval::sum_and_absolute_difference(&mut sum, &mut difference);
        prop_assert_eq!(negative, first > original);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&sum), original.add(first));
        let absolute = if original >= first { original.sub(first) } else { first.sub(original) };
        prop_assert_eq!(&InternalMpUint::from_limbs_slice(&difference), &absolute);
        SharedEval::overwrite_sum_with_absolute_difference(&mut sum, &mut source, negative);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&sum), absolute);

        let mut destination = vec![0; width.checked_add(shift).expect("shifted buffer fits")];
        // SAFETY: first.limbs().len()+shift <= width+shift, and the zero
        // destination retains two guard limbs above the complete shifted value.
        let frontier = unsafe { SharedEval::fused_add_shifted_in_place(&mut destination, first.limbs(), shift) };
        prop_assert!(frontier <= destination.len());
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&destination), first.shl(shift.checked_mul(LIMB_BITS).expect("shift fits")));
        SharedEval::add_coefficient_in_place(&mut destination, first.limbs(), shift);
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&destination), first.shl(shift.checked_mul(LIMB_BITS).expect("shift fits")).mul(&InternalMpUint::from_limb(2)));
    }
}
