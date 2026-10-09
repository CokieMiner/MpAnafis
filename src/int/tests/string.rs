//! Parsing and formatting contracts across radices, signs, and decimal widths.

extern crate std;

use std::panic::catch_unwind;

use alloc::{format, string::ToString};

use proptest::prelude::{any, prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

use super::strategies;

proptest! {
    #[test]
    fn formatting_and_parsing_preserve_values_across_radices(
        unsigned in strategies::uint(8), signed in strategies::int_maybe_bounded(8), selected_radix in 2_u32..=36,
    ) {
        for radix in [2, 8, 10, 16, 32, selected_radix] {
            let unsigned_digits = unsigned.to_string_radix(radix);
            let signed_digits = signed.to_string_radix(radix);
            prop_assert_eq!(&MpUint::from_str_radix(&unsigned_digits, radix).expect("valid encoding"), &unsigned);
            prop_assert_eq!(&MpInt::from_str_radix(&signed_digits, radix).expect("valid encoding"), &signed);
        }
        for (encoded, radix) in [(format!("{signed}"), 10), (format!("{signed:b}"), 2), (format!("{signed:o}"), 8), (format!("{signed:x}"), 16), (format!("{signed:X}"), 16)] {
            prop_assert_eq!(&MpInt::from_str_radix(&encoded, radix).expect("valid encoding"), &signed);
        }
        prop_assert_eq!(format!("{unsigned:x}"), unsigned.to_string_radix(16));
        prop_assert_eq!(unsigned.to_string().parse::<MpUint>().expect("valid decimal"), unsigned);
        let unlimited = MpInt::zero() + signed;
        let negative = -unlimited.abs();
        let magnitude = negative.unsigned_abs();
        let sign = if negative.is_zero() { "" } else { "-" };
        prop_assert_eq!(format!("{negative:#b}"), format!("{sign}0b{}", magnitude.to_string_radix(2)));
        prop_assert_eq!(format!("{negative:#o}"), format!("{sign}0o{}", magnitude.to_string_radix(8)));
        prop_assert_eq!(format!("{negative:#x}"), format!("{sign}0x{}", magnitude.to_string_radix(16)));
        prop_assert_eq!(format!("{negative:#X}"), format!("{sign}0x{}", magnitude.to_string_radix(16).to_ascii_uppercase()));
    }

    #[test]
    fn decimal_flags_and_leading_zero_parsing_match_native(unsigned in any::<u128>(), signed in any::<i128>()) {
        let u = MpUint::from(unsigned);
        let i = MpInt::from(signed);
        macro_rules! check_flags {
            ($value:ident, $scalar:ident) => {
                prop_assert_eq!(format!("{:+}", $value), format!("{:+}", $scalar));
                prop_assert_eq!(format!("{:08}", $value), format!("{:08}", $scalar));
                prop_assert_eq!(format!("{:+08}", $value), format!("{:+08}", $scalar));
                prop_assert_eq!(format!("{:>8}", $value), format!("{:>8}", $scalar));
                prop_assert_eq!(format!("{:^8}", $value), format!("{:^8}", $scalar));
                prop_assert_eq!(format!("{:.3}", $value), format!("{:.3}", $scalar));
            };
        }
        check_flags!(u, unsigned);
        check_flags!(i, signed);
        let padded_unsigned = format!("{unsigned:0>64}");
        prop_assert_eq!(padded_unsigned.parse::<MpUint>().expect("zero padding is valid"), u);
        let sign = if signed < 0 { "-" } else { "" };
        let padded_signed = format!("{sign}{:0>64}", signed.unsigned_abs());
        prop_assert_eq!(padded_signed.parse::<MpInt>().expect("zero padding is valid"), i);
    }
}

#[test]
fn partial_radix_digits_and_decimal_chunk_boundaries_preserve_padding() {
    // Radix 32 consumes five bits per digit; radix 8 consumes three.
    for (value, radix, expected) in [
        (1_u64 << 60, 32, "1000000000000"),
        (1_u64 << 61, 8, "200000000000000000000"),
        (u64::MAX, 32, "fvvvvvvvvvvvv"),
        (1_u64 << 39, 32, "g0000000"),
        (1_u64 << 23, 8, "40000000"),
    ] {
        assert_eq!(MpUint::from(value).to_string_radix(radix), expected);
    }
    for value in [
        0_u128,
        1,
        37,
        9999,
        10_u128.pow(18) - 1,
        10_u128.pow(18),
        10_u128.pow(19),
        10_u128.pow(19) + 1,
        10_u128.pow(38) - 1,
        10_u128.pow(38),
        u128::MAX,
    ] {
        let integer = MpUint::from(value);
        assert_eq!(integer.to_string(), value.to_string());
        assert_eq!(format!("{integer:+08}"), format!("{value:+08}"));
    }
    for value in [i128::MIN, -9999, -37, -1, 0, 1, 37, 9999, i128::MAX] {
        let integer = MpInt::from(value);
        assert_eq!(format!("{integer:+08}"), format!("{value:+08}"));
    }
    let small = (MpUint::one() << 512_u32) - MpUint::one();
    assert_eq!(
        small.to_string(),
        "1340780792994259709957402499820584612747936582059239337772356144372176403007354697\
6801874298166903427690031858186486050853753882811946569946433649006084095"
    );
    let wide = MpUint::one() << 3000_u32;
    let rendered = wide.to_string();
    assert_eq!(rendered.len(), 904);
    assert!(rendered.starts_with("12302319221611171769"));
    assert_eq!(rendered.parse::<MpUint>().expect("valid decimal"), wide);
}

#[test]
fn invalid_radices_and_zero_signs_have_consistent_boundaries() {
    for radix in [0, 1, 37, u32::MAX] {
        for value in [MpUint::zero(), MpUint::from(37_u8)] {
            assert!(catch_unwind(|| value.to_string_radix(radix)).is_err());
        }
        for value in [MpInt::zero(), MpInt::from(-37_i8)] {
            assert!(catch_unwind(|| value.to_string_radix(radix)).is_err());
        }
    }
    for representation in ["0", "+0", "-0", "000", "+000", "-000"] {
        let parsed = representation.parse::<MpInt>().expect("zero string parses");
        assert!(parsed.is_zero() && !parsed.is_negative() && !parsed.is_positive());
        let empty: [u8; 0] = [];
        assert_eq!(parsed.to_le_bytes(), empty);
    }
    for representation in ["0", "000", "000000"] {
        let parsed = representation
            .parse::<MpUint>()
            .expect("zero string parses");
        assert!(parsed.is_zero() && parsed.to_le_bytes().is_empty());
    }
}
