//! Endian conversion, significant-byte trimming, and storage transitions.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use crate::int::logic::unsigned::UintRepr;

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BYTES};

#[test]
fn byte_conversions_preserve_significant_bytes_and_select_storage_by_value() {
    let inline_bytes = INLINE_LIMBS
        .checked_mul(LIMB_BYTES)
        .expect("inline byte count");
    let check = |bytes: &[u8]| {
        let significant = bytes
            .iter()
            .rposition(|byte| *byte != 0)
            .map_or(0, |index| index.checked_add(1).expect("bounded byte count"));
        let reference = bytes.get(..significant).expect("significant prefix");
        let little = InternalMpUint::from_le_bytes(bytes);
        assert_eq!(little.to_le_bytes(), reference);
        assert_eq!(InternalMpUint::from_le_bytes(&little.to_le_bytes()), little);
        assert_eq!(InternalMpUint::from_be_bytes(&little.to_be_bytes()), little);
        assert_eq!(
            matches!(little.repr, UintRepr::Inline { .. }),
            significant <= inline_bytes
        );
        let reversed: alloc::vec::Vec<_> = bytes.iter().rev().copied().collect();
        let big = InternalMpUint::from_be_bytes(&reversed);
        assert_eq!(big, little);
        let expected: alloc::vec::Vec<_> = reference.iter().rev().copied().collect();
        assert_eq!(big.to_be_bytes(), expected);
        if let Some(native) = little.to_u128() {
            assert_eq!(InternalMpUint::from_u128(native), little);
        }
    };
    for length in 0..=inline_bytes.checked_add(2).expect("boundary width") {
        for padding in [0, 1, 17] {
            let mut bytes = vec![0xff; length];
            bytes.resize(length.checked_add(padding).expect("padded width"), 0);
            check(&bytes);
        }
    }
    let strategy = collection::vec(any::<u8>(), 0..=if cfg!(miri) { 40 } else { 1024 });
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |bytes| {
            check(&bytes);
            Ok(())
        })
        .expect("byte conversion property");
}
