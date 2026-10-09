//! Truncating quotient and remainder identities with signed overflow endpoints.

use proptest::test_runner::{Config, TestRunner};

use crate::int::{INLINE_LIMBS, InternalMpInt, LIMB_BITS};

use super::strategies::{public, signed, small_signed};

#[test]
fn division_variants_preserve_truncation_signs_and_reconstruct_the_dividend() {
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(small_signed(), small_signed()),
            |((a, left), (b, generated_divisor))| {
                let divisor = if b == 0 {
                    InternalMpInt::one()
                } else {
                    generated_divisor
                };
                let native_divisor = if b == 0 { 1 } else { i128::from(b) };
                let (quotient, remainder) = left.div_rem(&divisor);
                assert_eq!(
                    public(&quotient).to_i128(),
                    i128::from(a).checked_div(native_divisor)
                );
                assert_eq!(
                    public(&remainder).to_i128(),
                    i128::from(a).checked_rem(native_divisor)
                );
                check_division(&left, &divisor);
                Ok(())
            },
        )
        .expect("native truncating division agrees");
    cases
        .run(
            &(
                signed(if cfg!(miri) { 8 } else { 64 }),
                signed(if cfg!(miri) { 8 } else { 64 }),
            ),
            |(dividend, generated_divisor)| {
                let divisor = if generated_divisor.abs.is_zero() {
                    InternalMpInt::one()
                } else {
                    generated_divisor
                };
                check_division(&dividend, &divisor);
                check_division(&InternalMpInt::zero(), &divisor);
                if !dividend.abs.is_zero() {
                    check_division(&dividend, &dividend);
                }
                Ok(())
            },
        )
        .expect("wide quotient and remainder contracts agree");
    let inline_bits = LIMB_BITS
        .checked_mul(INLINE_LIMBS)
        .expect("small inline width");
    let minus_one = !InternalMpInt::zero();
    for bits in [
        1,
        LIMB_BITS.checked_sub(1).expect("nonzero limb width"),
        LIMB_BITS,
        LIMB_BITS.checked_add(1).expect("small width"),
        inline_bits.checked_sub(1).expect("nonzero inline width"),
        inline_bits,
        inline_bits.checked_add(1).expect("small width"),
    ] {
        let minimum = InternalMpInt::min_for_bits(bits);
        assert!(minimum.bounded_division_overflows(&minus_one, bits));
        assert!(!minimum.bounded_division_overflows(&InternalMpInt::one(), bits));
        assert!(
            !minimum
                .bounded_division_overflows(&minus_one, bits.checked_add(1).expect("small width"))
        );
        assert!(!InternalMpInt::max_for_bits(bits).bounded_division_overflows(&minus_one, bits));
        check_division(&minimum, &minus_one);
    }
}

fn check_division(dividend: &InternalMpInt, divisor: &InternalMpInt) {
    let (quotient, remainder) = dividend.div_rem(divisor);
    assert_eq!(quotient.mul(divisor).add(&remainder), *dividend);
    assert!(remainder.abs < divisor.abs);
    assert_eq!(
        quotient.is_positive,
        quotient.abs.is_zero() || dividend.is_positive == divisor.is_positive
    );
    assert_eq!(
        remainder.is_positive,
        remainder.abs.is_zero() || dividend.is_positive
    );
    assert_eq!(dividend.div(divisor), quotient);
    assert_eq!(dividend.rem(divisor), remainder);
    let mut in_place = dividend.clone();
    in_place.div_assign(divisor);
    assert_eq!(in_place, quotient);
    in_place.clone_from(dividend);
    in_place.rem_assign(divisor);
    assert_eq!(in_place, remainder);
}
