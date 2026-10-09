//! Minimal integer byte encodings, zero padding, and signed extension.

use alloc::vec::Vec;

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

proptest! {
    #[test]
    fn unsigned_bytes_match_native_and_ignore_zero_extensions(generated in any::<u128>()) {
        for value in [generated, 0, 1, 127, 128, 255, 256, u128::MAX] {
            let input = MpUint::from(value);
            let little = input.to_le_bytes();
            let big = input.to_be_bytes();
            let length = (128 - value.leading_zeros()).div_ceil(8) as usize;
            let native = value.to_le_bytes();
            prop_assert_eq!(little.as_slice(), native.get(..length).expect("native encoding span"));
            prop_assert!(little.iter().rev().eq(big.iter()));
            prop_assert_eq!(&MpUint::from_le_bytes(&little), &input);
            prop_assert_eq!(&MpUint::from_be_bytes(&big), &input);
            let mut extended = little;
            extended.resize(65, 0);
            prop_assert_eq!(&MpUint::from_le_bytes(&extended), &input);
            extended.reverse();
            prop_assert_eq!(MpUint::from_be_bytes(&extended), input);
        }
        for zero in [MpInt::from(0_i128), MpInt::from_le_bytes(&[0; 65]), MpInt::from_be_bytes(&[0; 65])] {
            prop_assert!(zero.is_zero() && !zero.is_negative() && !zero.is_positive());
            prop_assert!(zero.to_le_bytes().is_empty() && zero.to_be_bytes().is_empty());
        }
    }

    #[test]
    fn signed_bytes_match_native_twos_complement(generated in any::<i128>()) {
        for value in [generated, 0, 1, -1, 127, 128, -128, -129, 32768, -32769, i128::MIN, i128::MAX] {
            let input = MpInt::from(value);
            let little = input.to_le_bytes();
            let big = input.to_be_bytes();
            prop_assert!(little.iter().rev().eq(big.iter()));
            let significant = if value < 0 { !value } else { value }.cast_unsigned();
            let required = 128_u32.checked_sub(significant.leading_zeros())
                .and_then(|width| width.checked_add(1)).expect("at most 129 bits");
            let length = if value == 0 { 0 } else {
                usize::try_from(required.div_ceil(8)).expect("at most 16 bytes")
            };
            let native = value.to_le_bytes();
            prop_assert_eq!(little.as_slice(), native.get(..length).expect("native encoding span"));
            prop_assert_eq!(MpInt::from_le_bytes(&little).to_i128(), Some(value));
            prop_assert_eq!(MpInt::from_be_bytes(&big).to_i128(), Some(value));

            let mut extended = native.to_vec();
            let extension = if value < 0 { 0xFF } else { 0 };
            for width in [16, 17, 18, 63, 64, 65] {
                extended.resize(width, extension);
                let reversed: Vec<_> = extended.iter().rev().copied().collect();
                prop_assert_eq!(MpInt::from_le_bytes(&extended).to_i128(), Some(value));
                prop_assert_eq!(MpInt::from_be_bytes(&reversed).to_i128(), Some(value));
            }
        }
        prop_assert_eq!(MpInt::from_le_bytes(&[1, 0]).to_i128(), Some(1));
        prop_assert_eq!(MpInt::from_be_bytes(&[0xFF, 0x80]).to_i128(), Some(-128));
    }
}

#[test]
fn signed_byte_encoding_preserves_heap_carry_boundaries() {
    for bit in [255_usize, 256, 257, 319, 320, 321, 4095, 4096] {
        let magnitude = MpInt::one() << bit;
        for value in [
            &magnitude - MpInt::one(),
            magnitude.clone(),
            &magnitude + MpInt::one(),
        ] {
            for signed in [value.clone(), -value] {
                let little = signed.to_le_bytes();
                let big = signed.to_be_bytes();
                assert!(little.iter().rev().eq(big.iter()));
                assert_eq!(MpInt::from_le_bytes(&little), signed);
                assert_eq!(MpInt::from_be_bytes(&big), signed);
                if let [prefix @ .., extension] = little.as_slice()
                    && let Some(next) = prefix.last()
                {
                    assert!(!(*extension == 0 && next & 0x80 == 0));
                    assert!(!(*extension == 0xFF && next & 0x80 != 0));
                }
            }
        }
    }
}
