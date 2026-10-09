//! Minimal signed widths, endpoint construction, and two's-complement storage.

use alloc::vec;

use proptest::test_runner::{Config, TestRunner};

use crate::int::{INLINE_LIMBS, InternalMpInt, InternalMpUint, LIMB_BITS, Limb};

use super::strategies::signed;

#[test]
fn minimal_width_round_trips_and_limits_match_signed_powers() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(&signed(if cfg!(miri) { 8 } else { 64 }), |value| {
            let width = value.required_signed_bits_for_bounded_storage();
            assert!(width >= 1);
            assert_eq!(
                InternalMpInt::from_tc_bits(value.to_tc_bits(width), width),
                value
            );
            if width > 1 {
                let narrower = width.checked_sub(1).expect("width exceeds one");
                assert_ne!(
                    InternalMpInt::from_tc_bits(value.to_tc_bits(narrower), narrower),
                    value
                );
            }
            Ok(())
        })
        .expect("minimal signed widths are necessary and sufficient");

    let inline_bits = LIMB_BITS
        .checked_mul(INLINE_LIMBS)
        .expect("small inline width");
    for width in [
        1,
        2,
        LIMB_BITS.checked_sub(1).expect("limb width exceeds one"),
        LIMB_BITS,
        LIMB_BITS.checked_add(1).expect("small limb width"),
        inline_bits.checked_sub(1).expect("nonzero inline width"),
        inline_bits,
        inline_bits.checked_add(1).expect("small inline width"),
        4096,
        4097,
    ] {
        let magnitude = InternalMpUint::power_of_two(width.checked_sub(1).expect("positive width"));
        let minimum = InternalMpInt::min_for_bits(width);
        let maximum = InternalMpInt::max_for_bits(width);
        assert_eq!(minimum.abs, magnitude);
        assert!(!minimum.is_positive);
        assert_eq!(maximum.abs, magnitude.sub(&InternalMpUint::one()));
        assert!(maximum.is_positive);
        assert_eq!(maximum.abs.is_zero(), width == 1);
        for value in [&minimum, &maximum] {
            assert_eq!(value.required_signed_bits_for_bounded_storage(), width);
            assert_eq!(
                InternalMpInt::from_tc_bits(value.to_tc_bits(width), width),
                *value
            );
        }
        assert!(minimum.is_signed_min_for_width(width));
        assert!(!minimum.is_signed_min_for_width(width.checked_add(1).expect("small width")));
        assert!(!maximum.is_signed_min_for_width(width));
    }
}

#[test]
fn decoding_trims_sign_extension_without_losing_carries_or_owned_storage() {
    for limbs in [1_usize, 4, 5, 8, 64] {
        let full_width = limbs.checked_mul(LIMB_BITS).expect("small test width");
        for width in [
            full_width,
            full_width
                .checked_sub(3)
                .expect("one limb has at least 16 bits"),
        ] {
            let modulus = InternalMpUint::power_of_two(width);
            for offset in [
                0,
                width.div_euclid(2),
                width.checked_sub(1).expect("positive width"),
            ] {
                let expected = InternalMpUint::power_of_two(offset);
                let encoded = modulus.sub(&expected);
                let pointer = encoded.limbs().as_ptr();
                let capacity = encoded.capacity();
                let decoded = InternalMpInt::from_tc_bits(encoded, width);
                assert_eq!(decoded.abs, expected);
                assert!(!decoded.is_positive);
                if capacity > INLINE_LIMBS {
                    assert_eq!(decoded.abs.limbs().as_ptr(), pointer);
                    assert_eq!(decoded.abs.capacity(), capacity);
                }
            }
        }
    }
    for width in [
        LIMB_BITS
            .div_euclid(2)
            .checked_add(1)
            .expect("sub-limb width"),
        LIMB_BITS.checked_add(1).expect("small limb width"),
        LIMB_BITS
            .checked_mul(INLINE_LIMBS)
            .expect("small inline width"),
    ] {
        for is_positive in [false, true] {
            let value = InternalMpInt {
                abs: InternalMpUint::from_limbs(vec![Limb::MAX; 64]),
                is_positive,
            };
            let encoded = value.to_tc_bits(width);
            let expected = if is_positive {
                InternalMpUint::max_for_bits(width)
            } else {
                InternalMpUint::one()
            };
            assert_eq!(encoded, expected);
            assert_eq!(encoded.capacity(), INLINE_LIMBS);
        }
    }
}
