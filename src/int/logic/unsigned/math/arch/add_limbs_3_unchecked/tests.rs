//! Write-only addition and source-source aliasing.

#![expect(
    unsafe_code,
    reason = "Tests pass independent destinations and valid readable source spans"
)]

use core::mem::MaybeUninit;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::oracle::Oracle;

use super::{super::ArchKernels, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn addition_matches_recurrence_with_disjoint_and_overlapping_sources(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80)) {
        let (left, right): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&left, &right);
        check(&left, &left);
        if let Some((_, shifted)) = left.split_first() {
            check(left.get(..shifted.len()).expect("overlapping window"), shifted);
        }
    }
}

#[test]
fn carry_crosses_all_block_and_tail_boundaries() {
    for len in 0..=80 {
        let mut one = vec![0; len];
        if let Some(low) = one.first_mut() {
            *low = 1;
        }
        check(&vec![Limb::MAX; len], &one);
        check(&vec![Limb::MAX; len], &vec![Limb::MAX; len]);
    }
}

fn check(left: &[Limb], right: &[Limb]) {
    let len = left.len();
    let mut expected = left.to_vec();
    let expected_carry = Oracle::add(&mut expected, right);
    let mut actual =
        vec![MaybeUninit::<Limb>::uninit(); len.checked_add(2).expect("bounded guard width")];
    *actual.first_mut().expect("leading guard") = MaybeUninit::new(37);
    *actual.last_mut().expect("trailing guard") = MaybeUninit::new(37);
    // SAFETY: both sources cover len readable limbs and may alias each other.
    // The writable destination window is disjoint from both source spans.
    let carry = unsafe {
        ArchKernels::add_limbs_3_unchecked(
            actual.as_mut_ptr().cast::<Limb>().add(1),
            left.as_ptr(),
            right.as_ptr(),
            len,
        )
    };
    assert_eq!(carry, expected_carry);
    // SAFETY: the kernel initializes exactly len destination limbs, and both
    // adjacent guards were initialized before the call. MaybeUninit<Limb>
    // has Limb's layout; the shared view remains within the live allocation.
    let initialized =
        unsafe { core::slice::from_raw_parts(actual.as_ptr().cast::<Limb>(), actual.len()) };
    assert_eq!(initialized.get(1..=len).expect("result window"), expected);
    assert_eq!(initialized.first(), Some(&37));
    assert_eq!(initialized.last(), Some(&37));
    if len == 0 {
        assert_eq!(
            // SAFETY: none of the three pointers is accessed for an empty span.
            unsafe {
                ArchKernels::add_limbs_3_unchecked(
                    core::ptr::null_mut(),
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                )
            },
            0
        );
    }
}
