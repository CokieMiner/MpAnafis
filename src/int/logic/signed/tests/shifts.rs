//! Exact left shifts, floor-rounded right shifts, and bounded-width predicates.

use core::ops::{Shl, Shr};

use proptest::test_runner::{Config, TestRunner};

use crate::int::{InternalMpInt, LIMB_BITS};

use super::strategies::{public, signed, small_signed};

#[test]
fn shifts_match_native_arithmetic_and_preserve_signed_round_trips() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(small_signed(), 0_u32..=64, 0_u32..128),
            |((native, value), left_shift, right_shift)| {
                let a = i128::from(native);
                assert_eq!(
                    public(&Shl::shl(
                        &value,
                        usize::try_from(left_shift).expect("small shift")
                    ))
                    .to_i128(),
                    a.checked_shl(left_shift)
                );
                assert_eq!(
                    public(&Shr::shr(
                        &value,
                        usize::try_from(right_shift).expect("small shift")
                    ))
                    .to_i128(),
                    a.checked_shr(right_shift)
                );
                Ok(())
            },
        )
        .expect("native arithmetic shifts agree");
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                0_usize..=128,
                0_usize..=128,
            ),
            |(value, generated_shift, slack)| {
                let width = value.required_signed_bits_for_bounded_storage();
                let bits = width.checked_add(slack).expect("small generated width");
                for shift in [
                    0,
                    LIMB_BITS.checked_sub(1).expect("nonzero limb width"),
                    LIMB_BITS,
                    LIMB_BITS.checked_add(1).expect("small shift"),
                    width.checked_sub(1).expect("positive signed width"),
                    width,
                    width.checked_add(1).expect("small width"),
                    generated_shift,
                ] {
                    let shifted = Shl::shl(&value, shift);
                    let quotient = Shr::shr(&value, shift);
                    assert_eq!(Shr::shr(&shifted, shift), value);
                    assert_eq!(value.clone().mul_2exp(shift), shifted);
                    assert_eq!(Shl::shl(value.clone(), shift), shifted);
                    assert_eq!(Shr::shr(value.clone(), shift), quotient);
                    let mut in_place = value.clone();
                    in_place.shl_assign(shift);
                    assert_eq!(in_place, shifted);
                    in_place.clone_from(&value);
                    in_place.shr_assign(shift);
                    assert_eq!(in_place, quotient);
                    let divisor = public(&InternalMpInt::one())
                        .checked_shl(shift)
                        .expect("unlimited power of two");
                    assert_eq!(
                        public(&quotient),
                        public(&value)
                            .checked_div_floor(&divisor)
                            .expect("nonzero unlimited divisor")
                    );
                    assert_eq!(
                        value.bounded_shl_overflows(bits, shift),
                        !value.abs.is_zero() && shift > slack
                    );
                    assert!(quotient.is_positive || !quotient.abs.is_zero());
                }
                let collapsed = Shr::shr(&value, usize::MAX);
                assert_eq!(
                    collapsed,
                    if value.is_positive {
                        InternalMpInt::zero()
                    } else {
                        !InternalMpInt::zero()
                    }
                );
                Ok(())
            },
        )
        .expect("wide shifts agree across word and storage boundaries");
    let zero = InternalMpInt::zero();
    assert_eq!(Shl::shl(&zero, usize::MAX), zero);
    assert!(!zero.bounded_shl_overflows(1, usize::MAX));
}
