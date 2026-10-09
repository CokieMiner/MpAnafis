//! Decimal digit validation and native reciprocal division.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{Convert, DoubleLimb, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn decimal_chunks_validate_native_digits_and_division_reconstructs_values() {
    let divisor = Convert::DECIMAL_CHUNK_DIVISOR;
    let check = |limbs: &[Limb]| {
        let original = InternalMpUint::from_limbs_slice(limbs);
        let mut quotient = original.clone();
        let remainder = Convert::div_rem_decimal_chunk(&mut quotient);
        assert!(
            remainder < divisor,
            "decimal remainder is below the divisor"
        );
        assert_eq!(
            quotient
                .mul(&InternalMpUint::from_limb(divisor))
                .add(&InternalMpUint::from_limb(remainder)),
            original
        );
        if let Some(native) = original.to_u128() {
            let wide_divisor = u128::try_from(divisor).expect("native limb fits u128");
            assert_eq!(
                quotient.to_u128(),
                Some(native.checked_div(wide_divisor).expect("nonzero divisor"))
            );
            assert_eq!(
                u128::try_from(remainder).expect("native limb fits u128"),
                native.checked_rem(wide_divisor).expect("nonzero divisor")
            );
        }
    };
    for limbs in [
        vec![],
        vec![0],
        vec![divisor.checked_sub(1).expect("positive divisor")],
        vec![divisor],
        vec![0, divisor],
        vec![Limb::MAX; 5],
    ] {
        check(&limbs);
    }
    let mut runner = TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }));
    runner
        .run(&collection::vec(any::<Limb>(), 0..=16), |limbs| {
            check(&limbs);
            Ok(())
        })
        .expect("decimal division property");
    runner
        .run(
            &collection::vec(any::<u8>(), 1..=Convert::DECIMAL_CHUNK_DIGITS),
            |bytes| {
                let expected = bytes.iter().try_fold(0_usize, |value, &byte| {
                    let digit = byte.checked_sub(b'0')?;
                    if digit >= 10 {
                        return None;
                    }
                    value.checked_mul(10)?.checked_add(usize::from(digit))
                });
                assert_eq!(Convert::parse_decimal_chunk(&bytes).ok(), expected);
                let valid: alloc::vec::Vec<_> = bytes
                    .iter()
                    .map(|byte| b'0'.checked_add(byte.rem_euclid(10)).expect("decimal byte"))
                    .collect();
                let reference = valid.iter().fold(0_usize, |value, byte| {
                    value
                        .checked_mul(10)
                        .and_then(|prefix| {
                            prefix.checked_add(usize::from(
                                byte.checked_sub(b'0').expect("valid digit"),
                            ))
                        })
                        .expect("native decimal chunk")
                });
                assert_eq!(Convert::parse_decimal_chunk(&valid), Ok(reference));
                Ok(())
            },
        )
        .expect("decimal validation property");
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Exhaustively checks every byte in every decimal position; bounded validation and reconstruction properties run under Miri"
)]
fn decimal_validation_rejects_every_non_digit_at_every_native_position() {
    for width in 1..=Convert::DECIMAL_CHUNK_DIGITS {
        for fill in *b"019" {
            for position in 0..width {
                for candidate in 0_u8..=u8::MAX {
                    let mut input = vec![fill; width];
                    *input.get_mut(position).expect("digit position") = candidate;
                    let expected = input.iter().try_fold(0_usize, |value, &byte| {
                        let digit = byte.checked_sub(b'0')?;
                        if digit >= 10 {
                            return None;
                        }
                        value.checked_mul(10)?.checked_add(usize::from(digit))
                    });
                    assert_eq!(
                        Convert::parse_decimal_chunk(&input).ok(),
                        expected,
                        "bytes={input:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn normalized_small_division_matches_native_and_wide_reconstruction() {
    let strategy = (
        collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 32 }),
        1..=Limb::MAX,
    );
    let check = |limbs: &[Limb], divisor: Limb| {
        let original = InternalMpUint::from_limbs_slice(limbs);
        let mut quotient = original.clone();
        let normalized = DoubleLimb::try_from(divisor << divisor.leading_zeros())
            .expect("native divisor fits double limb");
        let reciprocal = DoubleLimb::MAX
            .checked_div(normalized)
            .expect("positive divisor")
            .checked_sub(1 << LIMB_BITS)
            .expect("normalized reciprocal has a leading limb");
        let remainder = Convert::div_rem_small(
            &mut quotient,
            divisor,
            Limb::try_from(reciprocal).expect("native reciprocal"),
        );
        assert!(remainder < divisor, "the remainder is below the divisor");
        assert_eq!(
            quotient
                .mul(&InternalMpUint::from_limb(divisor))
                .add(&InternalMpUint::from_limb(remainder)),
            original
        );
        if let Some(native) = original.to_u128() {
            let wide = u128::try_from(divisor).expect("native divisor");
            assert_eq!(
                quotient.to_u128(),
                Some(native.checked_div(wide).expect("positive divisor"))
            );
            assert_eq!(
                u128::try_from(remainder).expect("native remainder"),
                native.checked_rem(wide).expect("positive divisor")
            );
        }
    };
    for divisor in [1, 2, Convert::DECIMAL_CHUNK_DIVISOR, Limb::MAX] {
        for width in [0, 1, 3, 4, 5] {
            check(&vec![Limb::MAX; width], divisor);
        }
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, divisor)| {
            check(&limbs, divisor);
            Ok(())
        })
        .expect("small division property");
}
