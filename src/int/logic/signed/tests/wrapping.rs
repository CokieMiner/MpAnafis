//! Signed modular reduction, overflow reporting, and owned-buffer retention.

use alloc::vec;

use proptest::{
    prelude::any,
    test_runner::{Config, TestRunner},
};

use crate::int::{InternalMpInt, InternalMpUint, LIMB_BITS, Limb};

use super::strategies::{public, signed};

#[test]
fn wrapping_matches_sign_extension_and_wide_modular_congruence() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    for input in [i128::MIN, i128::MAX, -1, 0, 1] {
        for width in [
            1,
            2,
            LIMB_BITS.checked_sub(1).expect("limb width exceeds one"),
            LIMB_BITS,
            LIMB_BITS.checked_add(1).expect("native width is below 128"),
            127,
            128,
        ] {
            check_native_wrapping(input, width);
        }
    }
    cases
        .run(&(any::<i128>(), 1_usize..=128), |(input, width)| {
            check_native_wrapping(input, width);
            Ok(())
        })
        .expect("native modular representatives agree");
    cases
        .run(
            &(signed(if cfg!(miri) { 8 } else { 64 }), 1_usize..=512),
            |(value, width)| {
                let (wrapped, overflow) = value.clone().apply_wrapping_with_overflow(width);
                assert_eq!(overflow, value != wrapped);
                assert_eq!(value.clone().apply_wrapping(width), wrapped);
                assert!(wrapped.required_signed_bits_for_bounded_storage() <= width);
                assert!(wrapped.is_positive || !wrapped.abs.is_zero());
                let difference = value.sub(&wrapped);
                assert!(
                    difference
                        .abs
                        .rem(&InternalMpUint::power_of_two(width))
                        .is_zero()
                );
                Ok(())
            },
        )
        .expect("wide representatives fit and retain their residue class");
}

#[test]
fn overflowing_reduction_retains_owned_heap_capacity() {
    for limbs in [5_usize, 8, 64] {
        let width = limbs
            .checked_mul(LIMB_BITS)
            .and_then(|bits| bits.checked_sub(2))
            .expect("small heap width");
        for is_positive in [false, true] {
            let value = InternalMpInt {
                abs: InternalMpUint::from_limbs(vec![Limb::MAX; limbs]),
                is_positive,
            };
            let pointer = value.abs.limbs().as_ptr();
            let capacity = value.abs.capacity();
            let (wrapped, overflow) = value.apply_wrapping_with_overflow(width);
            assert!(overflow);
            assert!(wrapped.abs.is_one());
            assert_eq!(wrapped.is_positive, !is_positive);
            assert_eq!(wrapped.abs.capacity(), capacity);
            assert_eq!(wrapped.abs.limbs().as_ptr(), pointer);
        }
    }
}

fn check_native_wrapping(input: i128, width: usize) {
    let shift = u32::try_from(128_usize.checked_sub(width).expect("bounded width"))
        .expect("shift is below 128");
    let expected = input.wrapping_shl(shift).wrapping_shr(shift);
    let value = InternalMpInt {
        abs: InternalMpUint::from_u128(input.unsigned_abs()),
        is_positive: input >= 0,
    };
    let (wrapped, overflow) = value.clone().apply_wrapping_with_overflow(width);
    assert_eq!(public(&wrapped).to_i128(), Some(expected));
    assert_eq!(wrapped.is_positive, expected >= 0);
    assert_eq!(overflow, input != expected);
    assert_eq!(value.clone().apply_wrapping(width), wrapped);
    assert_eq!(
        InternalMpInt::from_tc_bits(value.to_tc_bits(width), width),
        wrapped
    );
    assert_eq!(
        wrapped.is_signed_min_for_width(width),
        expected == i128::MIN.wrapping_shr(shift)
    );
}
