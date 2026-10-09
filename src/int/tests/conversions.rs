//! Primitive integer and floating-point conversion domains.

use proptest::prelude::{any, prop_assert_eq, proptest};

use crate::{MpError, MpInt, MpUint};

proptest! {
    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Native floating-point casts provide independent ties-to-even conversion references"
    )]
    fn primitive_conversions_match_native_domains(unsigned in any::<u128>(), signed in any::<i128>()) {
        macro_rules! check_unsigned {
            ($($primitive:ty => $maximum:expr),+) => {$(
                for value in [unsigned, 0, 1, $maximum, u128::MAX] {
                    let expected = <$primitive>::try_from(value).map_err(|_| MpError::IntegerConversionLoss);
                    prop_assert_eq!(<$primitive>::try_from(MpUint::from(value)), expected);
                    prop_assert_eq!(<$primitive>::try_from(MpInt::from(value)), expected);
                    if let Ok(native) = <$primitive>::try_from(value) {
                        prop_assert_eq!(MpUint::from(native).to_u128(), Some(value));
                        prop_assert_eq!(MpInt::from(native).to_u128(), Some(value));
                    }
                }
                prop_assert_eq!(<$primitive>::try_from(MpInt::from(signed)), <$primitive>::try_from(signed).map_err(|_| MpError::IntegerConversionLoss));
            )+};
        }
        macro_rules! check_signed {
            ($($primitive:ty => ($minimum:expr, $maximum:expr)),+) => {$(
                for value in [signed, 0, -1, $minimum, $maximum, i128::MIN, i128::MAX] {
                    prop_assert_eq!(<$primitive>::try_from(MpInt::from(value)), <$primitive>::try_from(value).map_err(|_| MpError::IntegerConversionLoss));
                    if let Ok(native) = <$primitive>::try_from(value) {
                        prop_assert_eq!(MpInt::from(native).to_i128(), Some(value));
                        prop_assert_eq!(MpUint::try_from(native).and_then(|result| result.to_u128().ok_or(MpError::IntegerConversionLoss)),
                            u128::try_from(value).map_err(|_| MpError::NegativeInput));
                    }
                }
            )+};
        }
        check_unsigned!(
            u8 => u128::from(u8::MAX),
            u16 => u128::from(u16::MAX),
            u32 => u128::from(u32::MAX),
            u64 => u128::from(u64::MAX),
            usize => u128::try_from(usize::MAX).expect("native word fits u128")
        );
        check_signed!(
            i8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
            i16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
            i32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
            i64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
            isize => (
                i128::try_from(isize::MIN).expect("native word fits i128"),
                i128::try_from(isize::MAX).expect("native word fits i128")
            )
        );
        let u = MpUint::from(unsigned);
        let i = MpInt::from(signed);
        prop_assert_eq!(u128::try_from(u.clone()), Ok(unsigned));
        prop_assert_eq!(i128::try_from(i.clone()), Ok(signed));
        prop_assert_eq!(MpUint::try_from(signed).ok().and_then(|v| v.to_u128()), u128::try_from(signed).ok());
        prop_assert_eq!(u.to_u64(), u64::try_from(unsigned).ok());
        prop_assert_eq!(u.to_usize(), usize::try_from(unsigned).ok());
        prop_assert_eq!(u.to_i64(), i64::try_from(unsigned).ok());
        prop_assert_eq!(u.to_i128(), i128::try_from(unsigned).ok());
        prop_assert_eq!(u.to_isize(), isize::try_from(unsigned).ok());
        prop_assert_eq!(i.to_u64(), u64::try_from(signed).ok());
        prop_assert_eq!(u128::try_from(i.clone()).ok(), u128::try_from(signed).ok());
        prop_assert_eq!(i.to_u128(), u128::try_from(signed).ok());
        prop_assert_eq!(i.to_usize(), usize::try_from(signed).ok());
        prop_assert_eq!(i.to_i64(), i64::try_from(signed).ok());
        prop_assert_eq!(i.to_isize(), isize::try_from(signed).ok());
        prop_assert_eq!(MpUint::try_from(i.clone()).ok().and_then(|value| value.to_u128()), u128::try_from(signed).ok());
        let unsigned_f32 = unsigned as f32;
        let signed_f32 = signed as f32;
        prop_assert_eq!(u.to_f32().map(f32::to_bits), unsigned_f32.is_finite().then_some(unsigned_f32.to_bits()));
        prop_assert_eq!(i.to_f32().map(f32::to_bits), signed_f32.is_finite().then_some(signed_f32.to_bits()));
        prop_assert_eq!(u.to_f64().map(f64::to_bits), Some((unsigned as f64).to_bits()));
        prop_assert_eq!(i.to_f64().map(f64::to_bits), Some((signed as f64).to_bits()));
    }

}

#[test]
fn floating_point_overflow_is_decided_after_ties_to_even_rounding() {
    let max_f32 = ((MpUint::one() << 24_usize) - MpUint::one()) << 104_usize;
    let max_f64 = ((MpUint::one() << 53_usize) - MpUint::one()) << 971_usize;
    assert_eq!(max_f32.to_f32(), Some(f32::MAX));
    assert_eq!(max_f64.to_f64(), Some(f64::MAX));
    for (maximum, rounding_bit) in [(max_f32, 103_usize), (max_f64, 970_usize)] {
        let threshold = &maximum + (MpUint::one() << rounding_bit);
        let predecessor = &threshold - MpUint::one();
        let positive = MpInt::from(threshold.clone());
        if rounding_bit == 103 {
            assert_eq!(predecessor.to_f32(), Some(f32::MAX));
            assert_eq!(threshold.to_f32(), None);
            assert_eq!(positive.to_f32(), None);
            assert_eq!((-positive).to_f32(), None);
        } else {
            assert_eq!(predecessor.to_f64(), Some(f64::MAX));
            assert_eq!(threshold.to_f64(), None);
            assert_eq!(positive.to_f64(), None);
            assert_eq!((-positive).to_f64(), None);
        }
    }
}
