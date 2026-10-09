//! Signed width-limited kernels and finite or sign-extended bit searches.

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{BoundedPrecision, MpInt, Precision};

proptest! {
    #[test]
    fn signed_bit_operations_match_native_residues_and_sign_extension(
        value in any::<i128>(),
        bits in 1_u32..=128,
        rotation in any::<u32>(),
        bit in 0_u32..=130,
        enabled in any::<bool>(),
    ) {
        let width = usize::try_from(bits).expect("width at most 128");
        let precision = BoundedPrecision::new(width).expect("positive width");
        let padding = 128_u32.checked_sub(bits).expect("width at most 128");
        let mask = u128::MAX.wrapping_shr(padding);
        let residue = value.cast_unsigned() & mask;
        let decode = |encoded: u128| encoded.wrapping_shl(padding).cast_signed().wrapping_shr(padding);
        let rotate = rotation.rem_euclid(bits);
        let inverse = bits.checked_sub(rotate).expect("rotation below width");
        // A wrapping shift by 128 preserves the operand. Zero rotation yields residue | residue.
        let expected_left = (residue.wrapping_shl(rotate) | residue.wrapping_shr(inverse)) & mask;
        let expected_right = (residue.wrapping_shr(rotate) | residue.wrapping_shl(inverse)) & mask;
        let input = MpInt::from(value);
        for (outcome, expected) in [
            (input.rotate_left(rotation, width), expected_left),
            (input.rotate_right(rotation, width), expected_right),
            (input.reverse_bits(width), residue.reverse_bits().wrapping_shr(padding)),
            (input.not_with_width(width), !residue & mask),
        ] {
            let result = outcome.expect("validated width");
            prop_assert_eq!(result.to_i128(), Some(decode(expected)));
            prop_assert_eq!(result.precision(), Precision::Bounded(precision));
        }
        let bounded = MpInt::with_precision_wrapping(value, precision);
        prop_assert_eq!(bounded.try_not().expect("bounded width").to_i128(), Some(decode(!residue & mask)));
        prop_assert_eq!(bounded.count_ones(), Some(usize::try_from(residue.count_ones()).expect("at most 128 bits")));
        prop_assert_eq!(bounded.count_zeros(), Some(usize::try_from((!residue & mask).count_ones()).expect("at most 128 bits")));
        let scaled = residue.wrapping_shl(padding);
        prop_assert_eq!(bounded.leading_ones(), Some(usize::try_from(scaled.leading_ones().min(bits)).expect("at most 128 bits")));
        prop_assert_eq!(bounded.leading_zeros(), Some(usize::try_from(scaled.leading_zeros().min(bits)).expect("at most 128 bits")));
        prop_assert_eq!(bounded.trailing_ones(), Some(usize::try_from(residue.trailing_ones()).expect("at most 128 bits")));
        let from = usize::try_from(bit).expect("small bit");
        let changed = bounded.set_bit_to(from, enabled);
        let expected = if bit >= bits {
            residue
        } else if enabled {
            residue | 1_u128.wrapping_shl(bit)
        } else {
            residue & !1_u128.wrapping_shl(bit)
        };
        prop_assert_eq!(changed.to_i128(), Some(decode(expected)));
        prop_assert_eq!(changed.precision(), bounded.precision());
        let byte_shift = 16_u32.checked_sub(bits.div_ceil(8)).expect("at most sixteen bytes")
            .checked_mul(8).expect("at most 120 bits");
        let swapped = residue.swap_bytes().wrapping_shr(byte_shift) & mask;
        prop_assert_eq!(bounded.swap_bytes().expect("bounded width").to_i128(), Some(decode(swapped)));
        if bits.is_multiple_of(8) {
            prop_assert_eq!(bounded.swap_bytes().expect("bounded width").swap_bytes(), Some(bounded.clone()));
        }

        let native = value.cast_unsigned();
        let set = (from..128).find(|position| native & (1_u128 << position) != 0);
        let zero = (from..128).find(|position| native & (1_u128 << position) == 0);
        let unbounded_set = set.or_else(|| (value < 0).then_some(from.max(128)));
        let unbounded_zero = zero.unwrap_or_else(|| if value < 0 { usize::MAX } else { from.max(128) });
        prop_assert_eq!(input.find_next_set_bit(from), unbounded_set);
        prop_assert_eq!(input.find_next_zero_bit(from), unbounded_zero);
        prop_assert_eq!(bounded.find_next_set_bit(from), set.filter(|position| *position < width));
        prop_assert_eq!(bounded.find_next_zero_bit(from), zero.unwrap_or(width).min(width));
        for invalid_width in [0, usize::MAX] {
            prop_assert!(input.rotate_left(u32::MAX, invalid_width).is_none());
            prop_assert!(input.rotate_right(u32::MAX, invalid_width).is_none());
            prop_assert!(input.reverse_bits(invalid_width).is_none());
            prop_assert!(input.not_with_width(invalid_width).is_none());
        }
        prop_assert!(input.swap_bytes().is_none());
        prop_assert!(input.try_not().is_err());

        // Reversing 0x00FF in nine bits yields 0x0100, the signed value -256.
        let nine_bits = BoundedPrecision::new(9).expect("positive width");
        let boundary = MpInt::with_precision_checked(255_u16, nine_bits).expect("fits");
        let boundary_swapped = boundary.swap_bytes().expect("bounded precision");
        prop_assert_eq!(boundary_swapped.to_i128(), Some(-256));
        prop_assert_eq!(boundary_swapped.precision(), Precision::Bounded(nine_bits));
    }
}
