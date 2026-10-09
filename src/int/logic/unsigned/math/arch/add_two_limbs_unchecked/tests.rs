//! Independent addition chains and permitted exact aliases.

#![expect(
    unsafe_code,
    reason = "Tests pass disjoint writable spans and only permitted source aliases"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::oracle::Oracle;

use super::{super::ArchKernels, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn dual_addition_matches_recurrences_for_all_exact_alias_combinations(words in collection::vec((any::<Limb>(), any::<Limb>(), any::<Limb>(), any::<Limb>()), 0..=129)) {
        let initial_a = words.iter().map(|row| row.0).collect::<Vec<_>>();
        let initial_b = words.iter().map(|row| row.2).collect::<Vec<_>>();
        let source_a = words.iter().map(|row| row.1).collect::<Vec<_>>();
        let source_b = words.iter().map(|row| row.3).collect::<Vec<_>>();
        for shared_source in [false, true] {
            let right_source = if shared_source { &source_a } else { &source_b };
            for alias_a in [false, true] {
                for alias_b in [false, true] {
                    let mut a = if alias_a { source_a.clone() } else { initial_a.clone() };
                    let mut b = if alias_b { right_source.clone() } else { initial_b.clone() };
                    let mut expected_a = a.clone();
                    let mut expected_b = b.clone();
                    let carries = (Oracle::add(&mut expected_a, &source_a), Oracle::add(&mut expected_b, right_source));
                    let a_ptr = a.as_mut_ptr();
                    let b_ptr = b.as_mut_ptr();
                    let src_a_ptr = if alias_a { a_ptr.cast_const() } else { source_a.as_ptr() };
                    let src_b_ptr = if alias_b { b_ptr.cast_const() } else { right_source.as_ptr() };
                    // SAFETY: all spans cover words.len() limbs. Destinations are
                    // disjoint; each source is disjoint from destinations or
                    // exactly aliases its own destination. Source sharing is valid.
                    let actual = unsafe { ArchKernels::add_two_limbs_unchecked(a_ptr, src_a_ptr, b_ptr, src_b_ptr, words.len()) };
                    prop_assert_eq!(actual, carries);
                    prop_assert_eq!(a, expected_a);
                    prop_assert_eq!(b, expected_b);
                }
            }
        }
    }
}

#[test]
fn independent_carries_cross_block_boundaries_and_empty_spans_accept_null() {
    for len in 0..=80 {
        for right in [0, Limb::MAX] {
            let mut a = vec![Limb::MAX; len];
            let mut b = vec![right; len];
            let src = vec![Limb::MAX; len];
            let mut expected_a = a.clone();
            let mut expected_b = b.clone();
            let carries = (
                Oracle::add(&mut expected_a, &src),
                Oracle::add(&mut expected_b, &src),
            );
            // SAFETY: destinations are initialized and disjoint; the shared
            // immutable source covers len limbs and overlaps neither destination.
            let actual = unsafe {
                ArchKernels::add_two_limbs_unchecked(
                    a.as_mut_ptr(),
                    src.as_ptr(),
                    b.as_mut_ptr(),
                    src.as_ptr(),
                    len,
                )
            };
            assert_eq!(actual, carries);
            assert_eq!((a, b), (expected_a, expected_b));
        }
    }
    assert_eq!(
        // SAFETY: the empty contract accesses none of the four pointers.
        unsafe {
            ArchKernels::add_two_limbs_unchecked(
                core::ptr::null_mut(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null(),
                0,
            )
        },
        (0, 0)
    );
}
