//! ASCII domains, recursive split boundaries, and initialized product reconstruction.

use alloc::{format, string::String, vec, vec::Vec};

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use crate::error::ParseMpUintErrorKind;

use super::{Convert, INLINE_LIMBS, InternalMpUint, LIMB_BITS, RadixParameters};

fn assert_digit_accumulation(radix: u32, raw: &[u8]) {
    let mut expected = InternalMpUint::zero();
    let base = InternalMpUint::from_limb(usize::try_from(radix).expect("small radix"));
    let mut bytes = Vec::new();
    for (index, byte) in raw.iter().enumerate() {
        let digit = byte.rem_euclid(u8::try_from(radix).expect("small radix"));
        let encoded = Convert::byte_from_digit(digit);
        bytes.push(if index.is_multiple_of(2) {
            encoded.to_ascii_uppercase()
        } else {
            encoded
        });
        expected = expected
            .mul(&base)
            .add(&InternalMpUint::from_limb(usize::from(digit)));
    }
    let text = String::from_utf8(bytes.clone()).expect("ASCII digits");
    assert_eq!(
        InternalMpUint::from_str_radix(&text, radix),
        Ok(expected.clone())
    );
    assert_eq!(
        InternalMpUint::from_str_radix(&format!("000{text}"), radix),
        Ok(expected.clone())
    );
    if !radix.is_power_of_two() {
        let parameters = RadixParameters::for_limb(radix);
        for leaf in [1, 2, 3, 4, 8, 16, 32, 64] {
            for recursive in [false, true] {
                assert_eq!(
                    InternalMpUint::parse_non_power_of_two(
                        &bytes, radix, parameters, recursive, leaf
                    ),
                    Ok(expected.clone())
                );
            }
        }
    }
    for position in [
        0,
        bytes.len().div_euclid(2),
        bytes.len().checked_sub(1).expect("nonempty digits"),
    ] {
        let mut invalid = bytes.clone();
        *invalid.get_mut(position).expect("candidate position") = b'!';
        let input = String::from_utf8(invalid.clone()).expect("ASCII input");
        assert_eq!(
            InternalMpUint::from_str_radix(&input, radix)
                .expect_err("invalid digit")
                .kind(),
            &ParseMpUintErrorKind::InvalidDigit
        );
        if !radix.is_power_of_two() {
            for recursive in [false, true] {
                assert!(
                    InternalMpUint::parse_non_power_of_two(
                        &invalid,
                        radix,
                        RadixParameters::for_limb(radix),
                        recursive,
                        4
                    )
                    .is_err(),
                    "both parsing tiers reject every invalid digit"
                );
            }
        }
    }
}

#[test]
fn parsing_matches_digit_accumulation_and_rejects_invalid_input() {
    for radix in 2_u32..=36 {
        assert_digit_accumulation(radix, b"019azAZ");
        assert_eq!(
            InternalMpUint::from_str_radix("000", radix),
            Ok(InternalMpUint::zero())
        );
        for input in ["+1", " 1", "1 ", "1_0", "\u{e9}", "1\u{0}"] {
            assert_eq!(
                InternalMpUint::from_str_radix(input, radix)
                    .expect_err("invalid ASCII digit")
                    .kind(),
                &ParseMpUintErrorKind::InvalidDigit
            );
        }
        assert_eq!(
            InternalMpUint::from_str_radix("", radix)
                .expect_err("empty input")
                .kind(),
            &ParseMpUintErrorKind::Empty
        );
        assert_eq!(
            InternalMpUint::from_str_radix("-0", radix)
                .expect_err("negative input")
                .kind(),
            &ParseMpUintErrorKind::Negative
        );
    }
    for radix in [0, 1, 37, u32::MAX] {
        assert_eq!(
            InternalMpUint::from_str_radix("", radix)
                .expect_err("invalid radix")
                .kind(),
            &ParseMpUintErrorKind::InvalidRadix
        );
    }
    let strategy = (
        2_u32..=36,
        collection::vec(any::<u8>(), 1..=if cfg!(miri) { 8 } else { 256 }),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(radix, bytes)| {
            assert_digit_accumulation(radix, &bytes);
            Ok(())
        })
        .expect("parsing accumulation property");
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Configured parsing crossovers process hundreds of chunks in every radix; bounded validation and forced recursive reconstruction tests run under Miri"
)]
fn parsing_covers_inline_storage_and_dispatch_crossovers() {
    for radix in 2_u32..=36 {
        for bits in [
            1,
            LIMB_BITS,
            LIMB_BITS.checked_mul(INLINE_LIMBS).expect("inline width"),
        ] {
            let expected = InternalMpUint::max_for_bits(bits);
            let parsed = InternalMpUint::from_str_radix(&expected.to_string_radix(radix), radix)
                .expect("valid inline magnitude");
            assert_eq!(parsed, expected);
            assert_eq!(
                parsed.capacity(),
                INLINE_LIMBS,
                "radix={radix}, bits={bits}"
            );
        }
        let parameters = RadixParameters::for_limb(radix);
        let (entry, leaf) = if radix.is_power_of_two() {
            (3_usize, 1_usize)
        } else {
            Convert::parsing_thresholds(radix)
        };
        let crossover = parameters
            .max_digits
            .checked_mul(entry.checked_sub(1).expect("positive threshold"))
            .expect("bounded threshold");
        let leaf_digits = parameters
            .max_digits
            .checked_mul(leaf)
            .expect("bounded leaf");
        let lengths = [
            1,
            LIMB_BITS,
            leaf_digits.checked_sub(1).expect("positive width"),
            leaf_digits,
            leaf_digits.checked_add(1).expect("bounded width"),
            crossover.checked_sub(1).expect("positive threshold"),
            crossover,
            crossover.checked_add(1).expect("bounded threshold"),
            crossover
                .checked_mul(2)
                .and_then(|width| width.checked_add(1))
                .expect("bounded threshold"),
        ];
        let maximum = char::from(Convert::byte_from_digit(
            u8::try_from(radix.checked_sub(1).expect("positive radix")).expect("digit"),
        ));
        for length in lengths {
            let digits: String = core::iter::repeat_n(maximum, length).collect();
            let expected =
                InternalMpUint::from_limb(usize::try_from(radix).expect("radix fits Limb"))
                    .pow(u32::try_from(length).expect("test exponent"))
                    .sub(&InternalMpUint::one());
            for prefix in ["", "00000"] {
                assert_eq!(
                    InternalMpUint::from_str_radix(&format!("{prefix}{digits}"), radix),
                    Ok(expected.clone())
                );
            }
            assert!(
                InternalMpUint::from_str_radix(&format!("{digits}\u{e9}"), radix).is_err(),
                "the complete input must be validated"
            );
        }
    }
}

#[test]
fn recursive_products_initialize_compressed_powers_and_sparse_prefixes() {
    let chunks: &[usize] = if cfg!(miri) {
        &[7, 8, 9]
    } else {
        &[
            7, 11, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        ]
    };
    for radix in 3_u32..=36 {
        if radix.is_power_of_two() {
            continue;
        }
        let parameters = RadixParameters::for_limb(radix);
        let base = InternalMpUint::from_limb(usize::try_from(radix).expect("small radix"));
        let maximum = Convert::byte_from_digit(
            u8::try_from(radix.checked_sub(1).expect("positive radix")).expect("one digit"),
        );
        for &count in chunks {
            let width = count
                .checked_mul(parameters.max_digits)
                .expect("bounded input");
            let exponent = u32::try_from(width).expect("bounded exponent");
            let mut bytes = vec![maximum; width];
            let dense = base.pow(exponent).sub(&InternalMpUint::one());
            let sparse = base
                .pow(exponent.checked_sub(1).expect("nonempty input"))
                .add(&InternalMpUint::one());
            for expected in [dense, sparse] {
                for leaf in [1, 4, 16, 32, 64] {
                    assert_eq!(
                        InternalMpUint::parse_non_power_of_two(
                            &bytes, radix, parameters, true, leaf
                        ),
                        Ok(expected.clone()),
                        "radix={radix}, chunks={count}, leaf={leaf}"
                    );
                }
                assert_eq!(
                    InternalMpUint::from_str_radix(
                        core::str::from_utf8(&bytes).expect("ASCII digits"),
                        radix
                    ),
                    Ok(expected)
                );
                bytes.fill(b'0');
                *bytes.first_mut().expect("nonempty input") = b'1';
                *bytes.last_mut().expect("nonempty input") = b'1';
            }
        }
    }
}
