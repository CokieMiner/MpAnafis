//! Numeric ordering and representation-independent comparison contracts.

extern crate std;

use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};
use std::collections::hash_map::DefaultHasher;

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    prop_assert_eq, proptest,
};

use super::{InternalMpUint, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]
    #[test]
    fn comparison_matches_primitive_integer_order(left in any::<u128>(), right in any::<u128>()) {
        let left_value = InternalMpUint::from_u128(left);
        let right_value = InternalMpUint::from_u128(right);
        let expected = left.cmp(&right);
        prop_assert_eq!(left_value.cmp(&right_value), expected);
        prop_assert_eq!(left_value.partial_cmp(&right_value), Some(expected));
        prop_assert_eq!(left_value == right_value, left == right);
        prop_assert_eq!(
            InternalMpUint::cmp_limbs(left_value.limbs(), right_value.limbs()),
            expected
        );
        prop_assert_eq!(left_value.cmp(&left_value), Ordering::Equal);
    }

    #[test]
    fn padded_equal_width_windows_preserve_numeric_order(
        mut left in collection::vec(any::<Limb>(), 0..=129),
        mut right in collection::vec(any::<Limb>(), 0..=129),
        padding in 0_usize..=4,
    ) {
        let expected = InternalMpUint::from_limbs_slice(&left)
            .cmp(&InternalMpUint::from_limbs_slice(&right));
        let width = left.len().max(right.len()).checked_add(padding).expect("bounded fixture width");
        left.resize(width, 0);
        right.resize(width, 0);
        prop_assert_eq!(InternalMpUint::cmp_limbs(&left, &right), expected);
        prop_assert_eq!(InternalMpUint::cmp_limbs(&right, &left), expected.reverse());
        prop_assert_eq!(InternalMpUint::cmp_limbs(&left, &left), Ordering::Equal);
    }
}

#[test]
fn highest_differing_limb_dominates_lower_digits() {
    for width in [1_usize, 3, 4, 5, 64, 129] {
        for index in 0..width {
            let mut lower = vec![Limb::MAX; width];
            let mut greater = vec![0; width];
            lower
                .get_mut(index)
                .expect("fixture index is in bounds")
                .clone_from(&0);
            greater
                .get_mut(index)
                .expect("fixture index is in bounds")
                .clone_from(&1);
            for position in index.checked_add(1).expect("bounded fixture index")..width {
                lower
                    .get_mut(position)
                    .expect("fixture index is in bounds")
                    .clone_from(&0);
            }
            assert_eq!(InternalMpUint::cmp_limbs(&lower, &greater), Ordering::Less);
            assert_eq!(
                InternalMpUint::cmp_limbs(&greater, &lower),
                Ordering::Greater
            );
        }
    }
}

#[test]
fn comparisons_accept_overlapping_read_only_windows() {
    let limbs: [Limb; 3] = [0, 1, 0];
    let left = limbs.get(..2).expect("two-limb prefix");
    let right = limbs.get(1..).expect("two-limb suffix");
    assert_eq!(InternalMpUint::cmp_limbs(left, right), Ordering::Greater);
    assert_eq!(InternalMpUint::cmp_limbs(right, left), Ordering::Less);
    assert_eq!(InternalMpUint::cmp_limbs(left, left), Ordering::Equal);
    assert_eq!(InternalMpUint::cmp_limbs(&[], &[]), Ordering::Equal);
}

#[test]
fn allocation_and_inactive_inline_limbs_do_not_affect_comparison() {
    for width in [0_usize, 1, 3, 4, 5, 64, 129] {
        let limbs = vec![Limb::MAX; width];
        let mut original = InternalMpUint::from_limbs_slice(&limbs);
        if let crate::int::logic::unsigned::UintRepr::Inline { len, limbs: slots } =
            &mut original.repr
        {
            slots
                .get_mut(usize::from(*len)..)
                .expect("inactive inline suffix")
                .fill(17);
        }
        let capacity = width.checked_add(8).expect("bounded fixture capacity");
        let mut heap = InternalMpUint::with_capacity(capacity);
        heap.clone_from(&original);
        assert_eq!(original, heap);
        assert_eq!(original.cmp(&heap), Ordering::Equal);
        assert_eq!(heap.partial_cmp(&original), Some(Ordering::Equal));

        {
            let mut original_state = DefaultHasher::new();
            let mut heap_state = DefaultHasher::new();
            original.hash(&mut original_state);
            heap.hash(&mut heap_state);
            assert_eq!(original_state.finish(), heap_state.finish());
        }
    }
}
