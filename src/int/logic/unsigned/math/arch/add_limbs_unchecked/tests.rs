//! Addition contracts, exact aliasing, and carry propagation.

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
    fn addition_matches_recurrence_and_preserves_spans(words in collection::vec((any::<Limb>(), any::<Limb>()), 0..=80)) {
        let (initial, source): (Vec<_>, Vec<_>) = words.into_iter().unzip();
        check(&initial, &source);
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

fn check(initial: &[Limb], source: &[Limb]) {
    let len = source.len();
    let mut expected = initial.to_vec();
    let expected_carry = Oracle::add(&mut expected, source);
    let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
    actual
        .get_mut(1..=len)
        .expect("destination window")
        .copy_from_slice(initial);
    // SAFETY: the initialized len-limb window is disjoint from source.
    // Its two external canaries lie outside the span passed to the kernel.
    let carry = unsafe {
        ArchKernels::add_limbs_unchecked(actual.as_mut_ptr().add(1), source.as_ptr(), len)
    };
    assert_eq!(carry, expected_carry);
    assert_eq!(actual.get(1..=len).expect("result window"), expected);
    assert_eq!(actual.first(), Some(&37));
    assert_eq!(actual.last(), Some(&37));
    let mut doubled = source.to_vec();
    let mut expected_double = doubled.clone();
    let double_carry = Oracle::add(&mut expected_double, source);
    let pointer = doubled.as_mut_ptr();
    // SAFETY: both pointers come from the same writable span; exact aliasing
    // is permitted and each input limb is read before its output is stored.
    let actual_double_carry = unsafe { ArchKernels::add_limbs_unchecked(pointer, pointer, len) };
    assert_eq!(
        (doubled, actual_double_carry),
        (expected_double, double_carry)
    );
    if len == 0 {
        assert_eq!(
            // SAFETY: the empty contract returns before accessing either pointer.
            unsafe {
                ArchKernels::add_limbs_unchecked(core::ptr::null_mut(), core::ptr::null(), 0)
            },
            0
        );
    }
}
