//! Complete-product initialization, shape boundaries, and destination bounds.

#![expect(
    unsafe_code,
    reason = "The tests provide disjoint input spans and full writable product spans"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::{collection, prelude::*};

use crate::int::{
    logic::unsigned::math::arch::{ArchKernels, tests::oracle::Oracle},
    types::Limb,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]
    #[test]
    fn products_initialize_every_limb_and_match_schoolbook(
        left in collection::vec(any::<Limb>(), 2..=40),
        right in collection::vec(any::<Limb>(), 1..=40), dirty in any::<Limb>(),
    ) {
        check(&left, &right, dirty);
    }
}

#[test]
fn dense_products_cross_fixed_row_and_paired_row_boundaries() {
    for left_len in [2, 3, 4, 5, 8, 9, 13, 14, 17, 18, 31, 32, 33] {
        for right_len in [1, 2, 3, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 31, 32, 33] {
            if cfg!(miri) && (left_len > 5 || right_len > 5) {
                continue;
            }
            for (left_limb, right_limb) in [(Limb::MAX, Limb::MAX), (0, Limb::MAX), (Limb::MAX, 0)]
            {
                check(&vec![left_limb; left_len], &vec![right_limb; right_len], 37);
            }
        }
    }
}

fn check(left: &[Limb], right: &[Limb], dirty: Limb) {
    let width = left
        .len()
        .checked_add(right.len())
        .expect("bounded product width");
    let mut guarded = vec![dirty; width.checked_add(2).expect("bounded guard width")];
    // SAFETY: left has at least two readable limbs and right is nonempty,
    // and the disjoint destination window has left.len()+right.len() writable
    // limbs. The facade establishes target prerequisites.
    unsafe {
        ArchKernels::mul_basecase_unchecked(
            guarded.as_mut_ptr().add(1),
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
        );
    }
    assert_eq!(
        (guarded.first(), guarded.last()),
        (Some(&dirty), Some(&dirty))
    );
    let expected = Oracle::product(left, right);
    let end = width.checked_add(1).expect("bounded result end");
    assert_eq!(guarded.get(1..end).expect("product window"), expected);
    let mut uninitialized = vec![MaybeUninit::<Limb>::uninit(); width];
    // SAFETY: the same input bounds apply; the disjoint destination is aligned
    // and writable for the full product, which the kernel initializes.
    unsafe {
        ArchKernels::mul_basecase_unchecked(
            uninitialized.as_mut_ptr().cast(),
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
        );
    }
    for (actual, expected_limb) in uninitialized.iter().zip(&expected) {
        // SAFETY: the complete-product contract initializes every output limb.
        assert_eq!(unsafe { actual.assume_init() }, *expected_limb);
    }
}
