//! Signed radix parsing and minimal two's-complement encoding against GMP.

use mp_anafis::{MpInt, ParseMpIntErrorKind};
use rug::{Integer, integer::Order};

use crate::{assert_integer, float32, float64, signed_bytes};

pub fn fuzz_all(a: &MpInt, ra: &Integer, operation: u8, parameter: u16, bytes: &[u8]) {
    match operation % 5 {
        0 => {
            let radix = u32::from(parameter % 35) + 2;
            let text = a.to_string_radix(radix);
            assert_eq!(text, ra.to_string_radix(i32::try_from(radix).unwrap()));
            assert_integer(MpInt::from_str_radix(&text, radix).unwrap(), ra);
            let padded = format!(
                "{}00{}",
                if ra < &0 { "-" } else { "+" },
                ra.clone()
                    .abs()
                    .to_string_radix(i32::try_from(radix).unwrap())
                    .to_ascii_uppercase()
            );
            assert_integer(MpInt::from_str_radix(&padded, radix).unwrap(), ra);
        }
        1 => {
            let mut expected = signed_bytes(ra);
            assert_eq!(a.to_le_bytes(), expected);
            expected.reverse();
            assert_eq!(a.to_be_bytes(), expected);
            assert_integer(MpInt::from_le_bytes(&a.to_le_bytes()), ra);
            assert_integer(MpInt::from_be_bytes(&a.to_be_bytes()), ra);
            for (order, sign, actual) in [
                (Order::Lsf, bytes.last(), MpInt::from_le_bytes(bytes)),
                (Order::Msf, bytes.first(), MpInt::from_be_bytes(bytes)),
            ] {
                let mut expected = Integer::from_digits(bytes, order);
                if sign.is_some_and(|byte| byte & 0x80 != 0) {
                    expected -= Integer::from(1) << bytes.len().checked_mul(8).unwrap();
                }
                assert_integer(actual, &expected);
            }
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
            let magnitude = Integer::from(1) << (bits - 1);
            assert_integer(MpInt::max_for_precision(bits), &(magnitude.clone() - 1_u32));
            assert_integer(MpInt::min_for_precision(bits), &-magnitude);
        }
        _ => {
            for (text, radix, kind) in [
                ("", 10, ParseMpIntErrorKind::Empty),
                ("-", 10, ParseMpIntErrorKind::Empty),
                ("0", 1, ParseMpIntErrorKind::InvalidRadix),
                ("0", 37, ParseMpIntErrorKind::InvalidRadix),
                ("2", 2, ParseMpIntErrorKind::InvalidDigit),
                ("--1", 10, ParseMpIntErrorKind::InvalidDigit),
                (" 1", 10, ParseMpIntErrorKind::InvalidDigit),
            ] {
                let error = MpInt::from_str_radix(text, radix).unwrap_err();
                assert_eq!(error.kind(), &kind);
                assert_eq!(error.clone(), error);
                assert!(!error.to_string().is_empty());
                assert!(!format!("{error:?}").is_empty());
            }
            if let Ok(text) = core::str::from_utf8(bytes) {
                let radix = u32::from(parameter % 35) + 2;
                if let Ok(actual) = MpInt::from_str_radix(text, radix) {
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
