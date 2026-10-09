//! Primitive conversion domains and overflow bounds.

use proptest::{
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn native_conversions_preserve_values_and_reject_excess_width() {
    let check = |value: u128| {
        let unsigned = InternalMpUint::from_u128(value);
        assert_eq!(unsigned.to_u128(), Some(value));
        assert_eq!(unsigned.to_u64(), u64::try_from(value).ok());
        assert_eq!(unsigned.to_usize(), usize::try_from(value).ok());
        if let Ok(narrow) = u64::try_from(value) {
            assert_eq!(InternalMpUint::from_u64(narrow), unsigned);
            assert_eq!(InternalMpUint::from_u64(narrow).capacity(), INLINE_LIMBS);
        }
        let expected: alloc::vec::Vec<_> = value
            .to_le_bytes()
            .chunks(core::mem::size_of::<Limb>())
            .map(InternalMpUint::from_le_bytes)
            .collect();
        for (index, part) in expected.iter().enumerate() {
            assert_eq!(
                unsigned.limbs().get(index).copied().unwrap_or(0),
                part.to_usize().expect("one native limb")
            );
        }
    };
    for value in [0, 1, u128::from(u64::MAX), u128::MAX] {
        check(value);
    }
    for bit in [LIMB_BITS, 64, 128] {
        let boundary = InternalMpUint::power_of_two(bit);
        let below = boundary.sub(&InternalMpUint::one());
        match bit {
            64 => {
                assert_eq!(below.to_u64(), Some(u64::MAX));
                assert_eq!(boundary.to_u64(), None);
            }
            128 => {
                assert_eq!(below.to_u128(), Some(u128::MAX));
                assert_eq!(boundary.to_u128(), None);
            }
            _ => {
                assert_eq!(below.to_usize(), Some(usize::MAX));
                assert_eq!(boundary.to_usize(), None);
            }
        }
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&any::<u128>(), |value| {
            check(value);
            Ok(())
        })
        .expect("primitive conversion property");
}
