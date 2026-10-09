//! Limb inputs and public integer references for tuning runner tests.

use alloc::{vec, vec::Vec};

use proptest::{
    collection,
    prelude::{Just, Strategy, any, prop_oneof},
    sample::select,
};

use crate::{MpUint, tune_api::Limb};

/// Generates nonempty limb slices across inline, heap, dense, and zero-padded shapes.
pub fn limbs(max_len: usize) -> impl Strategy<Value = Vec<Limb>> {
    assert!(max_len != 0, "generated limb widths are nonzero");
    prop_oneof![
        select(vec![1, 4.min(max_len), 5.min(max_len), max_len]),
        1..=max_len,
    ]
    .prop_flat_map(|len| {
        prop_oneof![
            Just(vec![0; len]),
            Just(vec![Limb::MAX; len]),
            collection::vec(any::<Limb>(), len),
            (0..=len, any::<Limb>()).prop_map(move |(prefix, fill)| {
                let mut words = vec![0; len];
                words
                    .get_mut(..prefix)
                    .expect("the generated prefix fits the limb slice")
                    .fill(fill);
                words
            }),
        ]
    })
}

/// Constructs an unlimited public integer from little-endian native limbs.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Adding unlimited zero selects unlimited precision for the public reference value"
)]
pub fn integer(words: &[Limb]) -> MpUint {
    let byte_len = words
        .len()
        .checked_mul(size_of::<Limb>())
        .expect("test byte length fits usize");
    let mut bytes = Vec::with_capacity(byte_len);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    MpUint::zero() + MpUint::from_le_bytes(&bytes)
}
