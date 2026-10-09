//! Arbitrary operands around inline and nested-transform widths.

use super::*;

pub fn operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            prop::collection::vec(any::<Limb>(), 1..=4),
            prop::collection::vec(any::<Limb>(), 1..=4)
        ),
        (
            prop::collection::vec(any::<Limb>(), 5),
            prop::collection::vec(any::<Limb>(), 5)
        ),
        (
            prop::collection::vec(any::<Limb>(), 9),
            prop::collection::vec(any::<Limb>(), 7)
        ),
        (
            prop::collection::vec(any::<Limb>(), 17),
            prop::collection::vec(any::<Limb>(), 17)
        ),
    ]
}

pub fn transform_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    let minimum = SSA_BASE_MODULUS_BITS
        .div_euclid(LIMB_BITS)
        .div_euclid(2)
        .checked_add(1)
        .expect("test boundary fits");
    let maximum = minimum.checked_add(2).expect("test width fits");
    (
        prop::collection::vec(any::<Limb>(), minimum..=maximum),
        prop::collection::vec(any::<Limb>(), minimum..=maximum),
    )
}
