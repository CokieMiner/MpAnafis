//! Signed carry columns against an independent i128 recurrence.

use proptest::{
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use crate::int::logic::unsigned::math::gcd::reduction::subtract_products_with_carry;

use super::{ArchKernels, LIMB_BITS, Limb, SignedLimbCarry};

#[test]
fn signed_carry_columns_match_primitive_differences() {
    for (carry_low, carry_negative) in
        [(0, false), (Limb::MAX, false), (0, true), (Limb::MAX, true)]
    {
        for positive_value in [0, Limb::MAX] {
            check_column(
                positive_value,
                Limb::MAX,
                Limb::MAX,
                Limb::MAX,
                carry_low,
                carry_negative,
            );
        }
    }
    let columns = (
        any::<Limb>(),
        any::<Limb>(),
        any::<Limb>(),
        any::<Limb>(),
        any::<Limb>(),
        any::<bool>(),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) {
        4
    } else {
        1_024
    }))
    .run(&columns, |(a, b, c, d, carry, negative)| {
        check_column(a, b, c, d, carry, negative);
        Ok(())
    })
    .expect("signed carry property");
}

fn check_column(
    positive_value: Limb,
    positive_coefficient: Limb,
    negative_value: Limb,
    negative_coefficient: Limb,
    carry_low: Limb,
    carry_negative: bool,
) {
    let (actual_low, actual_carry) = subtract_products_with_carry(
        positive_value,
        positive_coefficient,
        negative_value,
        negative_coefficient,
        SignedLimbCarry {
            low: carry_low,
            negative: carry_negative,
        },
    );
    let (positive_low, positive_high) =
        ArchKernels::mul_limb_lo_hi(positive_value, positive_coefficient);
    let (negative_low, negative_high) =
        ArchKernels::mul_limb_lo_hi(negative_value, negative_coefficient);
    let [
        positive_lo_ref,
        positive_hi_ref,
        negative_lo_ref,
        negative_hi_ref,
        carry_ref,
        actual_carry_ref,
    ] = [
        positive_low,
        positive_high,
        negative_low,
        negative_high,
        carry_low,
        actual_carry.low,
    ]
    .map(|value| i128::try_from(value).expect("every supported limb fits i128"));
    let base = 1_i128 << LIMB_BITS;
    let carry_value = carry_ref
        .checked_sub(if carry_negative { base } else { 0 })
        .expect("signed carry fits i128");
    let low_difference = positive_lo_ref
        .checked_sub(negative_lo_ref)
        .and_then(|difference| difference.checked_add(carry_value))
        .expect("three signed limbs fit i128");
    let expected_low =
        Limb::try_from(low_difference.rem_euclid(base)).expect("the radix residue fits one limb");
    let expected_carry = positive_hi_ref
        .checked_sub(negative_hi_ref)
        .and_then(|difference| difference.checked_add(low_difference.div_euclid(base)))
        .expect("three signed limbs fit i128");
    let actual_carry_value = actual_carry_ref
        .checked_sub(if actual_carry.negative { base } else { 0 })
        .expect("signed carry fits i128");
    assert_eq!(actual_low, expected_low);
    assert_eq!(actual_carry_value, expected_carry);
    assert!(
        (base.checked_neg().expect("radix negation fits i128")..base).contains(&actual_carry_value)
    );
}
