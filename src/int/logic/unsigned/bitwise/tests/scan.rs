//! Population counts, width queries, searches, and implicit zero extension.

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{InternalMpUint, LIMB_BITS, Limb};

fn assert_scan_oracles(limbs: &[Limb], from: usize, width: usize) {
    let value = InternalMpUint::from_limbs_slice(limbs);
    let stored_bits = value
        .limbs()
        .len()
        .checked_mul(LIMB_BITS)
        .expect("bounded stored width");
    let first_set = (0..stored_bits).find(|bit| value.get_bit(*bit));
    let first_zero = (0..=stored_bits)
        .find(|bit| !value.get_bit(*bit))
        .expect("implicit zero bit");
    let next_set = (from..stored_bits).find(|bit| value.get_bit(*bit));
    let next_zero = (from..=stored_bits.max(from))
        .find(|bit| !value.get_bit(*bit))
        .expect("implicit zero bit");
    let significant = (0..stored_bits)
        .rev()
        .find(|bit| value.get_bit(*bit))
        .map_or(0, |bit| {
            bit.checked_add(1).expect("bounded significant width")
        });
    assert_eq!(value.significant_bits(), significant);
    assert_eq!(
        value.required_unsigned_bits_for_bounded_storage(),
        significant.max(1)
    );
    assert_eq!(
        value.count_ones(),
        (0..stored_bits).filter(|bit| value.get_bit(*bit)).count()
    );
    assert_eq!(
        value.count_zeros_for_width(width),
        (0..width).filter(|bit| !value.get_bit(*bit)).count()
    );
    assert_eq!(value.find_first_set_bit(), first_set);
    assert_eq!(value.trailing_zeros(), first_set.unwrap_or(0));
    assert_eq!(value.find_first_zero_bit(), first_zero);
    assert_eq!(value.trailing_ones(), first_zero);
    assert_eq!(value.find_next_set_bit(from), next_set);
    assert_eq!(value.find_next_zero_bit(from), next_zero);
    assert_eq!(value.find_next_set_bit(usize::MAX), None);
    assert_eq!(value.find_next_zero_bit(usize::MAX), usize::MAX);
    assert_eq!(
        value.leading_ones_for_width(width),
        (0..width)
            .rev()
            .take_while(|bit| value.get_bit(*bit))
            .count()
    );
    assert_eq!(
        value.leading_zeros_for_width(width),
        width.saturating_sub(significant)
    );
    assert_eq!(
        value.has_any_bits_set_below(width),
        (0..width).any(|bit| value.get_bit(bit))
    );
    assert_eq!(value.fits_in_bits(width), significant <= width);
    assert!(
        value.fits_in_bits(usize::MAX),
        "every fixture has addressable precision"
    );
    if significant != 0 {
        assert!(
            value <= InternalMpUint::max_for_bits(significant),
            "the reported width is sufficient"
        );
        assert!(
            value
                > InternalMpUint::max_for_bits(significant.checked_sub(1).expect("positive width")),
            "one fewer bit is insufficient"
        );
    }
}

#[test]
fn scans_and_precision_queries_match_individual_bit_oracles() {
    for width in [0, 1, 3, 4, 5, 12] {
        for bits in [
            0,
            1,
            LIMB_BITS,
            LIMB_BITS.checked_mul(width).expect("bounded window"),
            LIMB_BITS
                .checked_mul(width)
                .and_then(|bits| bits.checked_add(1))
                .expect("bounded window"),
        ] {
            assert_scan_oracles(&vec![Limb::MAX; width], bits, bits);
            assert_scan_oracles(&vec![0; width], bits, bits);
        }
    }
    for bits in [
        1,
        LIMB_BITS.checked_sub(1).expect("positive limb width"),
        LIMB_BITS.checked_add(1).expect("partial limb"),
        LIMB_BITS
            .checked_mul(5)
            .and_then(|width| width.checked_sub(1))
            .expect("heap run"),
    ] {
        let value = InternalMpUint::max_for_bits(bits);
        assert_scan_oracles(value.limbs(), bits, bits);
        assert_eq!(value.leading_ones_for_width(bits), bits);
        assert_eq!(value.leading_ones_for_width(usize::MAX), 0);
        assert!(
            value.has_any_bits_set_below(usize::MAX),
            "the finite prefix is nonzero"
        );
    }
    let max = LIMB_BITS.checked_mul(20).expect("bounded window");
    let strategy = (
        collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 16 }),
        0_usize..=max,
        0_usize..=max,
    );
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, from, width)| {
            assert_scan_oracles(&limbs, from, width);
            Ok(())
        })
        .expect("unsigned scan property");
}
