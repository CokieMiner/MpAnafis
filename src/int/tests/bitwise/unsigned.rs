//! Unsigned width-limited bit kernels match native residues.

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{BoundedPrecision, MpError, MpUint, Precision};

proptest! {
    #[test]
    fn unsigned_bit_kernels_match_native_residues(
        value in any::<u128>(), bits in 1_usize..=128,
        rotation in any::<u32>(), bit in 0_usize..=130, enabled in any::<bool>(),
    ) {
        let width = BoundedPrecision::new(bits).expect("positive width");
        let mask = u128::MAX >> (128 - bits);
        let residue = value & mask;
        let x = MpUint::with_precision_wrapping(value, width);
        let native_width = u32::try_from(bits).expect("width at most 128");
        let count = rotation % native_width;
        let inverse = native_width - count;
        let left = (residue.wrapping_shl(count) | residue.wrapping_shr(inverse)) & mask;
        let right = (residue.wrapping_shr(count) | residue.wrapping_shl(inverse)) & mask;
        for (outcome, expected) in [
            (x.rotate_left(rotation, bits), left),
            (x.rotate_right(rotation, bits), right),
            (x.reverse_bits(bits), residue.reverse_bits() >> (128 - bits)),
            (x.not_with_width(bits), !residue & mask),
        ] {
            let result = outcome.expect("valid width");
            prop_assert_eq!(result.to_u128(), Some(expected));
            prop_assert_eq!(result.precision(), Precision::Bounded(width));
        }
        prop_assert_eq!(x.rotate_left(rotation, bits).and_then(|v| v.rotate_right(rotation, bits)), Some(x.clone()));
        prop_assert_eq!(x.try_not().expect("bounded width").to_u128(), Some(!residue & mask));
        prop_assert_eq!(x.try_not().ok(), x.not_with_width(bits));
        prop_assert_eq!((MpUint::zero() + &x).try_not(), Err(MpError::WidthRequired));
        prop_assert_eq!(x.count_ones(), residue.count_ones() as usize);
        prop_assert_eq!(x.count_zeros(), Some(bits - residue.count_ones() as usize));
        let leading = usize::try_from((residue << (128 - bits)).leading_zeros().min(native_width))
            .expect("count at most 128");
        prop_assert_eq!(x.leading_zeros(), Some(leading));
        let changed = if bit >= bits { residue } else if enabled { residue | (1_u128 << bit) } else { residue & !(1_u128 << bit) };
        prop_assert_eq!(x.set_bit_to(bit, enabled).to_u128(), Some(changed));
        if !x.is_zero() { prop_assert!(x.get_bit(bits - 1 - leading)); }
        for invalid_width in [0, usize::MAX] {
            prop_assert!(x.rotate_left(rotation, invalid_width).is_none());
            prop_assert!(x.rotate_right(rotation, invalid_width).is_none());
            prop_assert!(x.reverse_bits(invalid_width).is_none());
            prop_assert!(x.not_with_width(invalid_width).is_none());
        }
    }
}
