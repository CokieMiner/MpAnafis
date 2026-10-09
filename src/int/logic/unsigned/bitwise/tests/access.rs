//! Individual-bit mutation and finite source-bounded range extraction.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn bit_access_and_ranges_preserve_unaffected_bits_and_normalization() {
    // A partial shift can remove the fifth limb after heap allocation.
    check_bit_access_and_ranges(
        &[0, 0, 0, 0, 1],
        LIMB_BITS.checked_sub(1).expect("positive limb width"),
        LIMB_BITS.checked_mul(6).expect("bounded range width"),
    );
    // Masking the fifth limb can leave four limbs or canonical zero.
    let masked_width = LIMB_BITS
        .checked_mul(INLINE_LIMBS)
        .and_then(|width| width.checked_add(1))
        .expect("partial fifth limb");
    for limbs in [
        [Limb::MAX, Limb::MAX, Limb::MAX, Limb::MAX, 2],
        [0, 0, 0, 0, 2],
    ] {
        check_bit_access_and_ranges(&limbs, 0, masked_width);
    }
    for width in [0, 1, 3, 4, 5, 8] {
        for start in [
            0,
            1,
            LIMB_BITS.checked_sub(1).expect("positive limb width"),
            LIMB_BITS,
        ] {
            for bits in [
                0,
                1,
                LIMB_BITS.checked_sub(1).expect("positive limb width"),
                LIMB_BITS,
                LIMB_BITS.checked_add(1).expect("bounded width"),
                LIMB_BITS.checked_mul(4).expect("inline width"),
                LIMB_BITS
                    .checked_mul(4)
                    .and_then(|span| span.checked_add(1))
                    .expect("heap transition"),
                LIMB_BITS.checked_mul(12).expect("bounded width"),
            ] {
                check_bit_access_and_ranges(&vec![Limb::MAX; width], start, bits);
            }
        }
    }
    let strategy = (
        collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 20 }),
        0_usize..=LIMB_BITS.checked_mul(24).expect("bounded start"),
        0_usize..=LIMB_BITS.checked_mul(28).expect("bounded width"),
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, start, bits)| {
            check_bit_access_and_ranges(&limbs, start, bits);
            Ok(())
        })
        .expect("bit access property");
}

fn check_bit_access_and_ranges(limbs: &[Limb], start: usize, bits: usize) {
    let value = InternalMpUint::from_limbs_slice(limbs);
    let end = start.checked_add(bits).expect("bounded bit range");
    let range = value.bit_range(start, end);
    assert_eq!(
        range,
        value.shr(start).rem(&InternalMpUint::power_of_two(bits))
    );
    let range_limb_bound = bits.div_ceil(LIMB_BITS).min(
        value
            .limbs()
            .len()
            .saturating_sub(start.div_euclid(LIMB_BITS)),
    );
    assert!(
        range.limbs().len() <= range_limb_bound,
        "the range stores only source-backed bits within its width"
    );
    assert!(
        range.limbs().last().is_none_or(|top| *top != 0),
        "the extracted magnitude is normalized"
    );
    // Normalization retains heap capacity; the initial width selects storage.
    if range_limb_bound <= INLINE_LIMBS {
        assert_eq!(range.capacity(), INLINE_LIMBS);
    }
    assert_eq!(value.bit_range(start, usize::MAX), value.shr(start));
    assert_eq!(value.bit_range(end, start), InternalMpUint::zero());
    let stored_bits = value
        .limbs()
        .len()
        .checked_mul(LIMB_BITS)
        .expect("bounded storage width");
    for bit in [0, start, stored_bits, stored_bits.saturating_sub(1)] {
        for set in [false, true] {
            let updated = value.set_bit_to(bit, set);
            let span = stored_bits.max(bit.checked_add(1).expect("bounded bit index"));
            assert_eq!(updated.get_bit(bit), set);
            for position in 0..span {
                if position != bit {
                    assert_eq!(updated.get_bit(position), value.get_bit(position));
                }
            }
            assert!(
                updated.limbs().last().is_none_or(|top| *top != 0),
                "the updated magnitude is normalized"
            );
        }
    }
    assert!(!value.get_bit(usize::MAX), "out-of-storage bits are zero");
    assert_eq!(value.set_bit_to(usize::MAX, false), value);
    assert!(
        value
            .bit_range(
                usize::MAX.checked_sub(1).expect("maximum index"),
                usize::MAX
            )
            .is_zero(),
        "a range beyond storage is zero"
    );
}
