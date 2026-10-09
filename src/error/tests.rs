//! Public parse-error classifications, formatting, and borrowed accessors.

#[cfg(feature = "std")]
use core::error::Error;

use alloc::{format, string::ToString};

#[cfg(feature = "std")]
use proptest::prop_assert;
use proptest::{
    arbitrary::any,
    prop_assert_eq, prop_oneof,
    test_runner::{Config, TestRunner},
};

#[cfg(feature = "std")]
use crate::PrecisionContext;
use crate::{
    MpInt, MpUint, ParseMpIntError, ParseMpIntErrorKind, ParseMpUintError, ParseMpUintErrorKind,
};

const SIGNED_EMPTY: &ParseMpIntErrorKind = ParseMpIntError::new(ParseMpIntErrorKind::Empty).kind();
const UNSIGNED_EMPTY: &ParseMpUintErrorKind =
    ParseMpUintError::new(ParseMpUintErrorKind::Empty).kind();

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "One property matrix covers both parsing error APIs without separate tests for each cause"
)]
fn parse_errors_preserve_causes_across_radices_signs_and_precision() {
    assert_eq!(SIGNED_EMPTY, &ParseMpIntErrorKind::Empty);
    assert_eq!(UNSIGNED_EMPTY, &ParseMpUintErrorKind::Empty);

    let strategy = (
        any::<u64>(),
        2_u32..=36,
        prop_oneof![0_u32..2, 37_u32..=u32::MAX],
    );
    let mut runner = TestRunner::new(Config {
        source_file: Some(file!()),
        ..Config::default()
    });
    runner
        .run(&strategy, |(magnitude, radix, invalid_radix)| {
            let valid = MpUint::from(magnitude).to_string_radix(radix);
            let invalid = format!("{valid}_");
            let negative = format!("-{valid}");
            for (text, base, kind, message) in [
                (
                    "",
                    radix,
                    ParseMpUintErrorKind::Empty,
                    "cannot parse integer from empty string",
                ),
                (
                    "",
                    invalid_radix,
                    ParseMpUintErrorKind::InvalidRadix,
                    "invalid radix",
                ),
                (
                    invalid.as_str(),
                    radix,
                    ParseMpUintErrorKind::InvalidDigit,
                    "invalid digit found in string",
                ),
                (
                    valid.as_str(),
                    invalid_radix,
                    ParseMpUintErrorKind::InvalidRadix,
                    "invalid radix",
                ),
                (
                    negative.as_str(),
                    radix,
                    ParseMpUintErrorKind::Negative,
                    "cannot parse unsigned integer from negative value",
                ),
            ] {
                let error = MpUint::from_str_radix(text, base).expect_err("invalid unsigned input");
                prop_assert_eq!(error.kind(), &kind);
                let cloned = error.clone();
                prop_assert_eq!(&cloned, &error);
                prop_assert_eq!(cloned.kind(), &kind);
                prop_assert_eq!(error.to_string(), message);
                #[cfg(feature = "std")]
                prop_assert!(Error::source(&error).is_none());
            }

            let negative_invalid = format!("-{invalid}");
            for (text, base, kind, message) in [
                (
                    "",
                    radix,
                    ParseMpIntErrorKind::Empty,
                    "cannot parse integer from empty string",
                ),
                (
                    "+",
                    radix,
                    ParseMpIntErrorKind::Empty,
                    "cannot parse integer from empty string",
                ),
                (
                    "-",
                    radix,
                    ParseMpIntErrorKind::Empty,
                    "cannot parse integer from empty string",
                ),
                (
                    "-",
                    invalid_radix,
                    ParseMpIntErrorKind::InvalidRadix,
                    "invalid radix",
                ),
                (
                    negative_invalid.as_str(),
                    radix,
                    ParseMpIntErrorKind::InvalidDigit,
                    "invalid digit found in string",
                ),
                (
                    valid.as_str(),
                    invalid_radix,
                    ParseMpIntErrorKind::InvalidRadix,
                    "invalid radix",
                ),
            ] {
                let error = MpInt::from_str_radix(text, base).expect_err("invalid signed input");
                prop_assert_eq!(error.kind(), &kind);
                let cloned = error.clone();
                prop_assert_eq!(&cloned, &error);
                prop_assert_eq!(cloned.kind(), &kind);
                prop_assert_eq!(error.to_string(), message);
                #[cfg(feature = "std")]
                prop_assert!(Error::source(&error).is_none());
            }

            for signs in ["+-", "-+", "--", "++"] {
                let text = format!("{signs}{valid}");
                let error = MpInt::from_str_radix(&text, radix).expect_err("repeated signs");
                prop_assert_eq!(error.kind(), &ParseMpIntErrorKind::InvalidDigit);
                prop_assert_eq!(error.to_string(), "invalid digit found in string");
            }

            #[cfg(feature = "std")]
            PrecisionContext::with_bounded(8, || {
                let unsigned = MpUint::from_str_radix("256", 10).expect_err("nine unsigned bits");
                prop_assert_eq!(unsigned.kind(), &ParseMpUintErrorKind::TooLarge);
                prop_assert_eq!(unsigned.to_string(), "value too large for parsing");
                for text in ["128", "-129"] {
                    let signed = MpInt::from_str_radix(text, 10).expect_err("nine signed bits");
                    prop_assert_eq!(signed.kind(), &ParseMpIntErrorKind::TooLarge);
                    prop_assert_eq!(signed.to_string(), "value too large for parsing");
                }
                Ok(())
            })?;
            Ok(())
        })
        .expect("public parse errors retain their classifications and formatting");
}
