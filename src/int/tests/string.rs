//! Property tests for public integer parsing and formatting.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "negation exercises the public signed formatting path on unlimited values"
)]

use alloc::{format, string::ToString};

use proptest::prelude::*;

use super::{std::panic::catch_unwind, strategies};
use crate::int::api::{ArbiInt, ArbiUint};

#[test]
fn to_string_radix_rejects_every_invalid_boundary_before_zero_fast_path() {
    let unsigned_zero = ArbiUint::zero();
    let unsigned_nonzero = ArbiUint::from(37_u8);
    let signed_zero = ArbiInt::zero();
    let signed_nonzero = ArbiInt::from(-37_i8);

    for radix in [0, 1, 37, u32::MAX] {
        assert!(catch_unwind(|| unsigned_zero.to_string_radix(radix)).is_err());
        assert!(catch_unwind(|| unsigned_nonzero.to_string_radix(radix)).is_err());
        assert!(catch_unwind(|| signed_zero.to_string_radix(radix)).is_err());
        assert!(catch_unwind(|| signed_nonzero.to_string_radix(radix)).is_err());
    }
}

#[test]
fn power_of_two_formatting_zero_extends_the_final_partial_digit() {
    // `2^60 = 32^12`: thirteen base-32 digits consume 65 digit bits from one
    // 64-bit limb. The final digit therefore needs one real bit plus four
    // implicit high zero bits, not a second limb.
    let value = ArbiUint::from(1_152_921_504_606_846_976_u64);
    assert_eq!(value.to_string_radix(32), "1000000000000");
}

#[test]
fn base8_formatting_partial_top_block() {
    // `2^61` has 62 significant bits: two full 24-bit blocks plus a top
    // block of 14 bits, which must render as a single leading digit
    // (python3: `format(2**61, 'o')`).
    let value = ArbiUint::from(2_305_843_009_213_693_952_u64);
    assert_eq!(value.to_string_radix(8), "200000000000000000000");
}

#[test]
fn base32_formatting_two_block_value() {
    // `u64::MAX` has 64 significant bits: a full 40-bit block of twelve
    // digits plus a 24-bit top block. Every lower digit is `v` (31) and the
    // top block renders as `f` (15) (python3: base32 of `2**64 - 1`).
    let value = ArbiUint::from(u64::MAX);
    assert_eq!(value.to_string_radix(32), "fvvvvvvvvvvvv");
}

#[test]
fn base32_formatting_exact_full_top_block() {
    // `2^39` has exactly 40 significant bits: one full block of eight
    // digits (python3: base32 of `2**39`).
    let value = ArbiUint::from(549_755_813_888_u64);
    assert_eq!(value.to_string_radix(32), "g0000000");
}

#[test]
fn base8_formatting_exact_full_top_block() {
    // `2^23` has exactly 24 significant bits: one full three-byte block of
    // eight digits (python3: `format(2**23, 'o')`).
    let value = ArbiUint::from(8_388_608_u64);
    assert_eq!(value.to_string_radix(8), "40000000");
}

#[test]
fn decimal_chunk_boundaries_are_intact() {
    // The radix-10 formatter divides by a single-limb power of ten (10^19 on
    // 64-bit limbs, 10^9 on 32-bit, 10^4 on 16-bit), so every power of ten is
    // a chunk boundary. These cases straddle the 64-bit boundaries exactly
    // (18/19/20 and 38/39 digit values) and prove the leading group is never
    // zero-padded. Expected strings come from python3.
    let below_19 = ArbiUint::from(999_999_999_999_999_999_u64);
    assert_eq!(below_19.to_string(), "999999999999999999");

    let at_10_18 = ArbiUint::from(1_000_000_000_000_000_000_u64);
    assert_eq!(at_10_18.to_string(), "1000000000000000000");

    let at_10_19 = ArbiUint::from(10_000_000_000_000_000_000_u128);
    assert_eq!(at_10_19.to_string(), "10000000000000000000");

    let just_above = ArbiUint::from(10_000_000_000_000_000_001_u128);
    assert_eq!(just_above.to_string(), "10000000000000000001");
}

#[test]
fn decimal_chunk_boundaries_for_wide_chunks() {
    // 10^38 and 10^38 - 1 straddle two 64-bit chunks of 19 digits each; the
    // lower group must keep its 19-digit zero padding.
    let below = ArbiUint::from(10_000_000_000_000_000_000_u128)
        * ArbiUint::from(10_000_000_000_000_000_000_u128)
        - ArbiUint::from(1_u8);
    assert_eq!(below.to_string(), "99999999999999999999999999999999999999");

    let exact = ArbiUint::from(10_000_000_000_000_000_000_u128)
        * ArbiUint::from(10_000_000_000_000_000_000_u128);
    assert_eq!(exact.to_string(), "100000000000000000000000000000000000000");

    let u128_max = ArbiUint::from(u128::MAX);
    assert_eq!(
        u128_max.to_string(),
        "340282366920938463463374607431768211455"
    );
}

#[test]
fn decimal_multilimb_small_and_recursive_paths() {
    // A 512-bit value uses the small direct path (8 limbs below the
    // machine-specific karatsuba threshold); a 3000-bit value forces the
    // recursive divide-and-conquer path. Both must render chunk-decimal
    // without leading zeros. Expected strings come from python3.
    let small = (ArbiUint::one() << 512_u32) - ArbiUint::one();
    assert_eq!(
        small.to_string(),
        "1340780792994259709957402499820584612747936582059239337772356144372176403007354697\
6801874298166903427690031858186486050853753882811946569946433649006084095"
    );

    let wide = ArbiUint::one() << 3000_u32;
    let rendered = wide.to_string();
    // Deterministic shape checks from python3: `2**3000` has 904 decimal
    // digits and starts with `12302319221611171769`.
    assert_eq!(rendered.len(), 904);
    assert!(
        rendered.starts_with("12302319221611171769"),
        "unexpected leading digits"
    );
    // Parse round trip proves the recursive divide-and-conquer formatting
    // reconstructs the exact value (in particular the zero-padded blocks).
    let round_trip = ArbiUint::from_str_radix(&rendered, 10).expect("parse radix 10");
    assert_eq!(round_trip, wide);
    assert_eq!(wide.to_string(), rendered);
}

proptest! {
    #[test]
    fn prop_roundtrip_base8_base32(value in strategies::uint(8)) {
        for radix in [8_u32, 32] {
            let digits = value.to_string_radix(radix);
            let round_trip = ArbiUint::from_str_radix(&digits, radix)
                .expect("parse power-of-two radix");
            prop_assert!(
                round_trip == value,
                "radix {} roundtrip failed",
                radix
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_string_format_roundtrip(value in strategies::int_maybe_bounded(8)) {
        let decimal = format!("{value}");
        let decimal_round_trip = ArbiInt::from_str_radix(&decimal, 10).expect("radix 10");
        prop_assert_eq!(&decimal_round_trip, &value, "radix 10");

        let binary = format!("{value:b}");
        let binary_round_trip = ArbiInt::from_str_radix(&binary, 2).expect("radix 2");
        prop_assert_eq!(&binary_round_trip, &value, "radix 2");

        let octal = format!("{value:o}");
        let octal_round_trip = ArbiInt::from_str_radix(&octal, 8).expect("radix 8");
        prop_assert_eq!(&octal_round_trip, &value, "radix 8");

        let lower_hex = format!("{value:x}");
        let lower_hex_round_trip = ArbiInt::from_str_radix(&lower_hex, 16).expect("radix 16");
        prop_assert_eq!(&lower_hex_round_trip, &value, "lower hex");

        let upper_hex = format!("{value:X}");
        let upper_hex_round_trip = ArbiInt::from_str_radix(&upper_hex, 16).expect("radix 16");
        prop_assert_eq!(&upper_hex_round_trip, &value, "upper hex");
    }

    #[test]
    fn prop_negative_alternate_formats_have_one_sign(
        magnitude in strategies::int_nonzero(8),
    ) {
        let negative = -magnitude.abs();

        let binary_digits = negative.value.abs.to_string_radix(2);
        let octal_digits = negative.value.abs.to_string_radix(8);
        let lower_hex_digits = negative.value.abs.to_string_radix(16);
        let upper_hex_digits = lower_hex_digits.to_ascii_uppercase();

        prop_assert_eq!(format!("{negative:#b}"), format!("-0b{binary_digits}"));
        prop_assert_eq!(format!("{negative:#o}"), format!("-0o{octal_digits}"));
        prop_assert_eq!(format!("{negative:#x}"), format!("-0x{lower_hex_digits}"));
        prop_assert_eq!(format!("{negative:#X}"), format!("-0x{upper_hex_digits}"));
    }
}

proptest! {
    #[test]
    fn prop_roundtrip_hex(value in strategies::uint(8)) {
        let hex = format!("{value:x}");
        if hex.is_empty() {
            return Ok(());
        }
        let round_trip = ArbiUint::from_str_radix(&hex, 16).expect("parse hex");
        prop_assert_eq!(round_trip, value, "hex roundtrip failed");
    }

    #[test]
    fn prop_roundtrip_decimal(value in strategies::uint(6)) {
        let decimal = value.to_string();
        let round_trip: ArbiUint = decimal.parse().expect("parse decimal");
        prop_assert_eq!(round_trip, value, "decimal roundtrip failed");
    }
}

proptest! {
    #[test]
    fn prop_string_parse_errors(
        magnitude in any::<u64>(),
        radix in 2_u32..=36,
        invalid_radix in prop_oneof![0_u32..2, 37_u32..=u32::MAX],
    ) {
        let empty_err = ArbiUint::from_str_radix("", 10).expect_err("should fail");
        prop_assert_eq!(
            empty_err.to_string(),
            "cannot parse integer from empty string"
        );

        let valid_digits = ArbiUint::from(magnitude).to_string_radix(radix);
        let invalid_digits = format!("{valid_digits}_");
        let char_err =
            ArbiUint::from_str_radix(&invalid_digits, radix).expect_err("should fail");
        prop_assert_eq!(char_err.to_string(), "invalid digit found in string");

        let radix_err =
            ArbiUint::from_str_radix(&valid_digits, invalid_radix).expect_err("should fail");
        prop_assert_eq!(radix_err.to_string(), "invalid radix");

        let negative_digits = format!("-{valid_digits}");
        let neg_err =
            ArbiUint::from_str_radix(&negative_digits, radix).expect_err("should fail");
        prop_assert_eq!(
            neg_err.to_string(),
            "cannot parse unsigned integer from negative value"
        );

        let empty_signed_err = ArbiInt::from_str_radix("-", 10).expect_err("should fail");
        prop_assert_eq!(
            empty_signed_err.to_string(),
            "cannot parse integer from empty string"
        );

        let invalid_signed_digits = format!("-{invalid_digits}");
        let invalid_char_int = ArbiInt::from_str_radix(&invalid_signed_digits, radix)
            .expect_err("should fail");
        prop_assert_eq!(
            invalid_char_int.to_string(),
            "invalid digit found in string"
        );
    }
}

proptest! {
    #[test]
    fn prop_string_parse_leading_zeros(value in any::<u64>()) {
        let encoded = format!("{:0>64}", value.to_string());
        let parsed = ArbiUint::from_str_radix(&encoded, 10).expect("should succeed");
        prop_assert_eq!(parsed, ArbiUint::from(value), "uint leading zeros parse failed for {}", value);
    }
}

proptest! {
    #[test]
    fn prop_string_parse_leading_zeros_signed(value in any::<i64>().prop_filter("non-min", |v| *v != i64::MIN)) {
        let prefix = if value >= 0 { "" } else { "-" };
        let encoded = format!("{}{:0>64}", prefix, value.abs());
        let parsed = ArbiInt::from_str_radix(&encoded, 10).expect("should succeed");
        prop_assert_eq!(parsed, ArbiInt::from(value), "signed leading zeros parse failed for {}", value);
    }
}
