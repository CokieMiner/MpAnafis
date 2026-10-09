//! Borrow propagation, stopping positions, and writable-span preservation.

#![expect(
    unsafe_code,
    reason = "The tests provide initialized writable spans and binary borrows"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use super::{super::ArchKernels, Limb};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn propagation_matches_recurrence_and_preserves_spans(
        source in collection::vec(any::<Limb>(), 0..=129), incoming in any::<bool>(),
    ) {
        check(&source, Limb::from(incoming));
    }
}

#[test]
fn borrow_stops_at_every_prefix_and_crosses_the_full_span() {
    for len in 0..=if cfg!(miri) { 9 } else { 129 } {
        for prefix in 0..=len {
            let source = (0..len)
                .map(|index| if index < prefix { 0 } else { 7 })
                .collect::<Vec<_>>();
            for incoming in [0, 1] {
                check(&source, incoming);
            }
        }
    }
}

fn check(source: &[Limb], incoming: Limb) {
    let mut expected = source.to_vec();
    let mut expected_borrow = incoming;
    for limb in &mut expected {
        let (value, underflow) = limb.overflowing_sub(expected_borrow);
        *limb = value;
        expected_borrow = Limb::from(underflow);
        if expected_borrow == 0 {
            break;
        }
    }
    let len = source.len();
    let end = len.checked_add(1).expect("bounded result end");
    let mut actual = vec![37; len.checked_add(2).expect("bounded guard width")];
    actual
        .get_mut(1..end)
        .expect("destination span")
        .copy_from_slice(source);
    // SAFETY: the guarded window contains len initialized writable limbs,
    // and incoming is either zero or one.
    let borrow = unsafe {
        ArchKernels::propagate_borrow_unchecked(actual.as_mut_ptr().add(1), len, incoming)
    };
    assert_eq!(
        (borrow, actual.get(1..end).expect("destination span")),
        (expected_borrow, expected.as_slice())
    );
    assert_eq!((actual.first(), actual.last()), (Some(&37), Some(&37)));
    if len == 0 {
        assert_eq!(
            // SAFETY: the empty path returns the binary incoming borrow
            // without accessing the pointer.
            unsafe { ArchKernels::propagate_borrow_unchecked(core::ptr::null_mut(), 0, incoming) },
            incoming
        );
    }
}
