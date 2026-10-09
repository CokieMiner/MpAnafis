//! Large conversion fixtures crossing recursive radix and storage boundaries.

use mp_anafis::{MpInt, MpUint};
use rug::{Integer, integer::Order};

use crate::assert_integer;
#[cfg(feature = "num-traits")]
use crate::{signed, unsigned};

#[test]
#[cfg_attr(
    miri,
    ignore = "Large conversion comparisons use GMP native FFI unavailable to Miri"
)]
fn radix_and_byte_conversions_match_gmp_across_boundaries() {
    for bytes in [0, 1, 8, 32, 33, 257, 2048, 4096, 32768] {
        let mut state = 17_u8;
        let input: Vec<_> = (0..bytes)
            .map(|_| {
                state = state.wrapping_mul(157).wrapping_add(1);
                state
            })
            .collect();
        let reference = Integer::from_digits(&input, Order::Lsf);
        for radix in 2_u32..=36 {
            let digits = reference.to_string_radix(i32::try_from(radix).unwrap());
            let parsed = MpUint::from_str_radix(&digits, radix).unwrap();
            assert_eq!(parsed.to_le_bytes(), reference.to_digits::<u8>(Order::Lsf));
            assert_eq!(parsed.to_string_radix(radix), digits);
            let negative = format!("-{digits}");
            let parsed = MpInt::from_str_radix(&negative, radix).unwrap();
            assert_integer(&parsed, &-reference.clone());
            assert_eq!(
                parsed.to_string_radix(radix),
                if reference == 0 {
                    "0".to_owned()
                } else {
                    negative
                }
            );
        }
        for order in [Order::Lsf, Order::Msf] {
            let reference = Integer::from_digits(&input, order);
            let parsed = if order == Order::Lsf {
                MpUint::from_le_bytes(&input)
            } else {
                MpUint::from_be_bytes(&input)
            };
            assert_eq!(parsed.to_le_bytes(), reference.to_digits::<u8>(Order::Lsf));
        }
    }
    #[cfg(feature = "num-traits")]
    for value in [
        0.0_f64,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        i64::MIN as f64,
        2_f64.powi(63),
        2_f64.powi(64),
        2_f64.powi(127),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        let mut input = vec![10, 4, 0, 0, 0, 0, 255];
        input.extend(value.to_bits().to_be_bytes());
        unsigned::run(&input);
        signed::run(&input);
        let mut input = vec![10, 4, 0, 0, 0, 0, 255];
        // The f32 cast selects the corresponding finite, overflow, or NaN category.
        input.extend((value as f32).to_bits().to_be_bytes());
        unsigned::run(&input);
        signed::run(&input);
    }
}
