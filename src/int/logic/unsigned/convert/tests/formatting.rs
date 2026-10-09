//! Radix formatting domains, finite digit windows, and reusable writer state.

use core::fmt::{Error as FmtError, Result as FmtResult, Write};

use alloc::{
    format,
    string::{String, ToString},
    vec,
};

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{Convert, FormatCache, InternalMpUint, LIMB_BITS, RadixParameters};

struct RejectingWriter {
    remaining: usize,
    accepted: String,
}

impl Write for RejectingWriter {
    fn write_str(&mut self, text: &str) -> FmtResult {
        if self.remaining == 0 {
            return Err(FmtError);
        }
        self.remaining = self.remaining.checked_sub(1).expect("available write");
        self.accepted.push_str(text);
        Ok(())
    }
}

#[test]
fn formatted_digits_cover_native_values_and_partial_binary_windows() {
    for bits in 1_usize
        ..=LIMB_BITS
            .checked_mul(5)
            .and_then(|width| width.checked_add(1))
            .expect("bounded width")
    {
        for (radix, digit_bits) in [(2_u32, 1_usize), (4, 2), (8, 3), (16, 4), (32, 5)] {
            let width = bits.div_ceil(digit_bits);
            let top_bits = bits
                .checked_sub(1)
                .expect("positive width")
                .rem_euclid(digit_bits)
                .checked_add(1)
                .expect("one digit");
            let maximum_digit = char::from(Convert::byte_from_digit(
                u8::try_from((1_usize << top_bits).checked_sub(1).expect("positive mask"))
                    .expect("one digit"),
            ));
            let sparse_digit = char::from(Convert::byte_from_digit(
                u8::try_from(1_usize << top_bits.checked_sub(1).expect("positive top width"))
                    .expect("one digit"),
            ));
            let lower = char::from(Convert::byte_from_digit(
                u8::try_from(radix.checked_sub(1).expect("positive radix")).expect("one digit"),
            ));
            let suffix = width.checked_sub(1).expect("positive digit count");
            for (value, expected) in [
                (
                    InternalMpUint::max_for_bits(bits),
                    format!("{maximum_digit}{}", lower.to_string().repeat(suffix)),
                ),
                (
                    InternalMpUint::power_of_two(bits.checked_sub(1).expect("positive width")),
                    format!("{sparse_digit}{}", "0".repeat(suffix)),
                ),
            ] {
                assert_eq!(value.to_string_radix(radix), expected);
                assert_eq!(InternalMpUint::from_str_radix(&expected, radix), Ok(value));
            }
        }
    }
    let strategy = (
        collection::vec(any::<u8>(), 0..=if cfg!(miri) { 24 } else { 2048 }),
        2_u32..=36,
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(bytes, radix)| {
            let value = InternalMpUint::from_le_bytes(&bytes);
            let digits = value.to_string_radix(radix);
            assert!(
                !digits.is_empty(),
                "formatting always emits at least one digit"
            );
            assert!(
                digits == "0" || !digits.starts_with('0'),
                "only canonical zero starts with zero"
            );
            assert_eq!(
                InternalMpUint::from_str_radix(&digits, radix),
                Ok(value.clone())
            );
            if let Some(mut native) = value.to_u128() {
                let mut reference = vec![];
                loop {
                    reference.push(Convert::byte_from_digit(
                        u8::try_from(
                            native
                                .checked_rem(u128::from(radix))
                                .expect("positive radix"),
                        )
                        .expect("one digit"),
                    ));
                    native = native
                        .checked_div(u128::from(radix))
                        .expect("positive radix");
                    if native == 0 {
                        break;
                    }
                }
                reference.reverse();
                assert_eq!(digits.as_bytes(), reference);
            }
            Ok(())
        })
        .expect("radix formatting property");
}

#[test]
fn recursive_cache_preserves_padding_across_growth_shrinkage_and_writer_errors() {
    let mut cache = FormatCache::new();
    let one = InternalMpUint::one();
    for radix in 3_u32..=36 {
        if radix.is_power_of_two() {
            continue;
        }
        let parameters = RadixParameters::for_limb(radix);
        let base = InternalMpUint::from_limb(parameters.max_power);
        let maximum = char::from(Convert::byte_from_digit(
            u8::try_from(radix.checked_sub(1).expect("positive radix")).expect("one digit"),
        ));
        let mut cases = vec![
            (InternalMpUint::zero(), String::from("0")),
            (one.clone(), String::from("1")),
        ];
        for blocks in [1_u32, 2, 4, 8, 16] {
            let width = parameters
                .max_digits
                .checked_mul(usize::try_from(blocks).expect("small exponent"))
                .expect("bounded output");
            let power = base.pow(blocks);
            cases.extend([
                (power.sub(&one), maximum.to_string().repeat(width)),
                (power.clone(), format!("1{}", "0".repeat(width))),
                (
                    power.add(&one),
                    format!(
                        "1{}1",
                        "0".repeat(width.checked_sub(1).expect("positive width"))
                    ),
                ),
            ]);
        }
        for (value, expected) in cases.iter().chain(cases.iter().rev()).chain(cases.iter()) {
            let mut output = String::new();
            value
                .format_recursive_writer_with_cache(radix, &mut output, &mut cache)
                .expect("String writes are infallible");
            assert_eq!(&output, expected, "radix={radix}");
        }
        let value = base.square().sub(&one);
        let mut writer = RejectingWriter {
            remaining: 1,
            accepted: String::new(),
        };
        assert!(
            value
                .format_recursive_writer_with_cache(radix, &mut writer, &mut cache)
                .is_err(),
            "writer errors propagate"
        );
        assert_eq!(
            writer.accepted,
            maximum.to_string().repeat(parameters.max_digits)
        );
        let mut output = String::new();
        value
            .format_recursive_writer_with_cache(radix, &mut output, &mut cache)
            .expect("the cache remains reusable after failure");
        assert_eq!(
            output,
            maximum.to_string().repeat(
                parameters
                    .max_digits
                    .checked_mul(2)
                    .expect("bounded output")
            )
        );
    }
}
