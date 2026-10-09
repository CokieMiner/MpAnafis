//! Write-only subtraction and source-source aliasing.

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
    fn subtraction_matches_recurrence_with_disjoint_and_identical_sources(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80)) {
        let (left, right): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&left, &right);
        check(&left, &left);
    }
}

#[test]
fn borrow_crosses_all_block_and_tail_boundaries() {
    for len in 0..=if cfg!(miri) { 17 } else { 80 } {
        let mut one = vec![0; len];
        if let Some(low) = one.first_mut() {
            *low = 1;
        }
        check(&vec![0; len], &one);
        check(&vec![0; len], &vec![Limb::MAX; len]);
    }
}

fn check(left: &[Limb], right: &[Limb]) {
    let len = left.len();
    let mut expected = left.to_vec();
    let expected_borrow = Oracle::sub(&mut expected, right);
    let mut uninitialized = vec![MaybeUninit::<Limb>::uninit(); len];
    // SAFETY: both aligned sources cover len initialized limbs and may alias.
    // The disjoint aligned output covers len writable uninitialized limbs;
    // the complete subtraction initializes every output limb.
    let uninitialized_borrow = unsafe {
        ArchKernels::sub_limbs_3_unchecked(
            uninitialized.as_mut_ptr().cast(),
            left.as_ptr(),
            right.as_ptr(),
            len,
        )
    };
    assert_eq!(uninitialized_borrow, expected_borrow);
    for (limb, expected_limb) in uninitialized.iter().zip(&expected) {
        // SAFETY: the complete subtraction initialized each output limb.
        assert_eq!(unsafe { limb.assume_init() }, *expected_limb);
    }
    let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
    // SAFETY: both sources cover len readable limbs and may alias each other.
    // The writable destination window is disjoint from both source spans.
    let borrow = unsafe {
        ArchKernels::sub_limbs_3_unchecked(
            actual.as_mut_ptr().add(1),
            left.as_ptr(),
            right.as_ptr(),
            len,
        )
    };
    assert_eq!(borrow, expected_borrow);
    assert_eq!(actual.get(1..=len).expect("result window"), expected);
    assert_eq!(actual.first(), Some(&37));
    assert_eq!(actual.last(), Some(&37));
    if len == 0 {
        assert_eq!(
            // SAFETY: none of the three pointers is accessed for an empty span.
            unsafe {
                ArchKernels::sub_limbs_3_unchecked(
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
