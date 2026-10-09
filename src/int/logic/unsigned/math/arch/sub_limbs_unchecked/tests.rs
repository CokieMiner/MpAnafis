//! Subtraction contracts, exact aliasing, and borrow propagation.

#![expect(
    unsafe_code,
    reason = "Tests pass owned, aligned spans and the permitted exact alias"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::int::logic::unsigned::math::arch::tests::oracle::Oracle;

use super::{super::ArchKernels, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn subtraction_matches_recurrence_and_preserves_spans(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80)) {
        let (initial, source): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&initial, &source);
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

fn check(initial: &[Limb], source: &[Limb]) {
    let len = source.len();
    let mut expected = initial.to_vec();
    let expected_borrow = Oracle::sub(&mut expected, source);
    let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
    actual
        .get_mut(1..=len)
        .expect("destination window")
        .copy_from_slice(initial);
    // SAFETY: the initialized len-limb window is disjoint from source.
    // Its two external canaries lie outside the span passed to the kernel.
    let borrow = unsafe {
        ArchKernels::sub_limbs_unchecked(actual.as_mut_ptr().add(1), source.as_ptr(), len)
    };
    assert_eq!(borrow, expected_borrow);
    assert_eq!(actual.get(1..=len).expect("result window"), expected);
    assert_eq!(actual.first(), Some(&37));
    assert_eq!(actual.last(), Some(&37));
    let mut aliased = source.to_vec();
    let pointer = aliased.as_mut_ptr();
    // SAFETY: exact source/destination aliasing is permitted for this complete
    // initialized span. Subtracting a span from itself gives zero and no borrow.
    let alias_borrow = unsafe { ArchKernels::sub_limbs_unchecked(pointer, pointer, len) };
    assert_eq!(alias_borrow, 0);
    assert!(aliased.iter().all(|limb| *limb == 0));
    if len == 0 {
        assert_eq!(
            // SAFETY: the empty contract returns before accessing either pointer.
            unsafe {
                ArchKernels::sub_limbs_unchecked(core::ptr::null_mut(), core::ptr::null(), 0)
            },
            0
        );
    }
}
