//! Radix, byte, primitive, and malformed-input conversions against GMP.

use mp_anafis::{MpUint, ParseMpUintErrorKind};
use rug::{Integer, integer::Order};

use crate::{assert_integer, float32, float64};

pub fn fuzz_all(a: &MpUint, ra: &Integer, operation: u8, parameter: u16, bytes: &[u8]) {
    match operation % 5 {
        0 => {
            let radix = u32::from(parameter % 35) + 2;
            let text = a.to_string_radix(radix);
            assert_eq!(text, ra.to_string_radix(i32::try_from(radix).unwrap()));
            assert_integer(MpUint::from_str_radix(&text, radix).unwrap(), ra);
            assert_integer(
                MpUint::from_str_radix(&format!("00{}", text.to_ascii_uppercase()), radix).unwrap(),
                ra,
            );
        }
        1 => {
            assert_eq!(a.to_le_bytes(), ra.to_digits::<u8>(Order::Lsf));
            assert_eq!(a.to_be_bytes(), ra.to_digits::<u8>(Order::Msf));
            assert_integer(MpUint::from_le_bytes(&a.to_le_bytes()), ra);
            assert_integer(MpUint::from_be_bytes(&a.to_be_bytes()), ra);
            assert_integer(
                MpUint::from_le_bytes(bytes),
                &Integer::from_digits(bytes, Order::Lsf),
            );
            assert_integer(
                MpUint::from_be_bytes(bytes),
                &Integer::from_digits(bytes, Order::Msf),
            );
        }
        2 => {
            assert_eq!(a.to_u64(), ra.to_u64());
            assert_eq!(a.to_u128(), ra.to_u128());
            assert_eq!(a.to_usize(), ra.to_usize());
            assert_eq!(a.to_i64(), ra.to_i64());
            assert_eq!(a.to_i128(), ra.to_i128());
            assert_eq!(a.to_isize(), ra.to_isize());
            assert_eq!(a.to_f64(), float64(ra));
            assert_eq!(a.to_f32(), float32(ra));
        }
        3 => {
            let bits = usize::from(parameter % 2047) + 1;
            assert_integer(
                MpUint::max_for_precision(bits),
                &((Integer::from(1) << bits) - 1_u32),
            );
            assert_integer(MpUint::min_for_precision(bits), &Integer::new());
        }
        _ => {
            for (text, radix, kind) in [
                ("", 10, ParseMpUintErrorKind::Empty),
                ("0", 1, ParseMpUintErrorKind::InvalidRadix),
                ("0", 37, ParseMpUintErrorKind::InvalidRadix),
                ("2", 2, ParseMpUintErrorKind::InvalidDigit),
                ("-1", 10, ParseMpUintErrorKind::Negative),
                ("+1", 10, ParseMpUintErrorKind::InvalidDigit),
                (" 1", 10, ParseMpUintErrorKind::InvalidDigit),
            ] {
                let error = MpUint::from_str_radix(text, radix).unwrap_err();
                assert_eq!(error.kind(), &kind);
                assert_eq!(error.clone(), error);
                assert!(!error.to_string().is_empty());
                assert!(!format!("{error:?}").is_empty());
            }
            if let Ok(text) = core::str::from_utf8(bytes) {
                let radix = u32::from(parameter % 35) + 2;
                let actual = MpUint::from_str_radix(text, radix);
                if let Ok(actual) = actual {
                    let expected = Integer::from_str_radix(
                        text.trim_start_matches('+'),
                        i32::try_from(radix).unwrap(),
                    )
                    .unwrap();
                    assert_integer(actual, &expected);
                }
            }
        }
    }
}
