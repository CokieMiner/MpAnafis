//! Canonical signed inputs and unlimited public numeric references.

use alloc::vec;

use proptest::{
    collection,
    prelude::{Just, Strategy, any, prop_oneof},
    sample::select,
};

use crate::{
    MpInt, MpUint,
    int::{InternalMpInt, InternalMpUint, Limb},
};

pub fn signed(max_limbs: usize) -> impl Strategy<Value = InternalMpInt> {
    assert!(max_limbs != 0, "test limb limit is nonzero");
    let widths = prop_oneof![
        select(vec![
            0,
            1,
            3.min(max_limbs),
            4.min(max_limbs),
            5.min(max_limbs),
            max_limbs
        ]),
        0..=max_limbs,
    ];
    (widths, any::<bool>()).prop_flat_map(|(width, positive)| {
        prop_oneof![
            Just(vec![0; width]),
            Just(vec![Limb::MAX; width]),
            collection::vec(any::<Limb>(), width),
            (0..=width, any::<Limb>()).prop_map(move |(prefix, fill)| {
                let mut limbs = vec![0; width];
                limbs
                    .get_mut(..prefix)
                    .expect("generated prefix fits")
                    .fill(fill);
                limbs
            }),
        ]
        .prop_map(move |limbs| {
            let abs = InternalMpUint::from_limbs(limbs);
            InternalMpInt {
                is_positive: positive || abs.is_zero(),
                abs,
            }
        })
    })
}

pub fn small_signed() -> impl Strategy<Value = (i64, InternalMpInt)> {
    prop_oneof![any::<i64>(), select(vec![i64::MIN, i64::MAX, -1, 0, 1]),].prop_map(|native| {
        let abs = InternalMpUint::from_u128(u128::from(native.unsigned_abs()));
        (
            native,
            InternalMpInt {
                abs,
                is_positive: native >= 0,
            },
        )
    })
}

pub fn public(value: &InternalMpInt) -> MpInt {
    let magnitude = MpUint::zero()
        .checked_add(&MpUint::from_le_bytes(&value.abs.to_le_bytes()))
        .expect("an unlimited magnitude accepts every test input");
    let signed_value = MpInt::from(magnitude);
    if value.is_positive {
        signed_value
    } else {
        MpInt::zero()
            .checked_sub(&signed_value)
            .expect("unlimited negation is representable")
    }
}
