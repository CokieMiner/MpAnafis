//! Canonical construction, initialized growth, allocation reuse, and raw write ownership.

#![expect(
    unsafe_code,
    reason = "Tests supply normalized limb vectors and fully initialize reserved raw spans before their explicit length commitments"
)]

#[cfg(feature = "std")]
use core::panic::AssertUnwindSafe;
#[cfg(feature = "std")]
use std::panic::catch_unwind;

use alloc::vec;

use proptest::{
    collection,
    prelude::{ProptestConfig, any},
    test_runner::TestRunner,
};

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};

const EMPTY_LIMBS: [Limb; 0] = [];

#[test]
fn constructors_assignment_and_normalization_preserve_the_active_magnitude() {
    let check = |input: &[Limb], padding: usize| {
        let mut expected = input.to_vec();
        while expected.last() == Some(&0) {
            let _removed = expected.pop();
        }
        let mut padded = input.to_vec();
        padded.resize(
            input
                .len()
                .checked_add(padding)
                .expect("bounded test length"),
            0,
        );
        let owned = InternalMpUint::from_limbs(padded.clone());
        let borrowed = InternalMpUint::from_limbs_slice(&padded);
        // SAFETY: removing every high zero leaves a normalized prefix, including empty zero.
        let normalized = unsafe { InternalMpUint::from_limbs_normalized(expected.clone()) };
        for value in [&owned, &borrowed, &normalized] {
            assert_eq!(value.limbs(), expected);
            assert_eq!(
                matches!(value.repr, UintRepr::Inline { .. }),
                expected.len() <= INLINE_LIMBS
            );
            let mut repeated = value.clone();
            repeated.normalize();
            assert_eq!(repeated, *value);
        }
        for capacity in [0, 32] {
            let mut assigned = InternalMpUint::with_capacity(capacity);
            let pointer = assigned.limbs().as_ptr();
            assigned.clone_from_slice(&padded);
            assert_eq!(assigned.limbs(), expected);
            if capacity != 0 {
                assert_eq!(assigned.limbs().as_ptr(), pointer);
                assert_eq!(assigned.capacity(), capacity);
            }
            let extracted = core::array::from_fn::<_, INLINE_LIMBS, _>(|index| {
                expected.get(index).copied().unwrap_or(0)
            });
            assert_eq!(assigned.extract_4(), extracted);
            assigned.set_limb(7);
            assert_eq!(assigned.limbs(), [7]);
            assigned.set_limb(0);
            assert_eq!(assigned.limbs(), EMPTY_LIMBS);
        }
    };
    for width in [0, 1, 3, 4, 5, 12] {
        for padding in [0, 1, 6] {
            check(&vec![Limb::MAX; width], padding);
            check(&vec![0; width], padding);
        }
    }
    assert_eq!(InternalMpUint::zero().limbs(), EMPTY_LIMBS);
    assert_eq!(InternalMpUint::one().limbs(), [1]);
    assert_eq!(InternalMpUint::from_limbs_2(1, 0).limbs(), [1]);
    assert_eq!(InternalMpUint::from_limbs_4(1, 2, 0, 0).limbs(), [1, 2]);
    let strategy = (collection::vec(any::<Limb>(), 0..=12), 0_usize..=6);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, padding)| {
            check(&limbs, padding);
            Ok(())
        })
        .expect("canonical storage property");
}

#[test]
fn cloning_clearing_and_swapping_preserve_owned_allocations() {
    for width in 0..=INLINE_LIMBS.checked_add(2).expect("heap boundary") {
        let mut source = InternalMpUint::with_capacity(32);
        source.clone_from_slice(&vec![Limb::MAX; width]);
        let source_pointer = source.limbs().as_ptr();
        let cloned = source.clone();
        assert_eq!(cloned, source);
        assert_eq!(
            matches!(cloned.repr, UintRepr::Inline { .. }),
            width <= INLINE_LIMBS
        );
        assert_eq!(source.limbs().as_ptr(), source_pointer);
        assert_eq!(source.capacity(), 32);
        let mut inline = InternalMpUint::one();
        inline.clone_from(&source);
        assert_eq!(inline, source);
        assert_eq!(
            matches!(inline.repr, UintRepr::Inline { .. }),
            width <= INLINE_LIMBS
        );
        let mut heap = InternalMpUint::with_capacity(64);
        let heap_pointer = heap.limbs().as_ptr();
        heap.clone_from(&source);
        assert_eq!(heap, source);
        assert_eq!(heap.limbs().as_ptr(), heap_pointer);
        assert_eq!(heap.capacity(), 64);
        source.swap(&mut heap);
        assert_eq!(source.limbs().as_ptr(), heap_pointer);
        assert_eq!(heap.limbs().as_ptr(), source_pointer);
        heap.clear();
        assert_eq!(heap.limbs(), EMPTY_LIMBS);
        assert_eq!(heap.capacity(), 32);
        assert_eq!(heap.limbs().as_ptr(), source_pointer);
        source.shrink_to_fit();
        assert_eq!(source.limbs(), vec![Limb::MAX; width]);
    }
}

#[test]
fn inactive_inline_slots_remain_hidden_and_newly_exposed_slots_are_zeroed() {
    let original = [1, 2, 3, 4];
    for retained in 0..=INLINE_LIMBS {
        for assign in [false, true] {
            let mut value = InternalMpUint::from_limbs_4(1, 2, 3, 4);
            let prefix = original.get(..retained).expect("retained prefix");
            if assign {
                value.clone_from_slice(prefix);
            } else {
                value.resize(retained);
            }
            assert_eq!(value.limbs(), prefix);
            assert!(
                value
                    .extract_4()
                    .get(retained..)
                    .expect("inactive suffix")
                    .iter()
                    .all(|limb| *limb == 0),
                "extraction hides inactive slots"
            );
            value.resize(INLINE_LIMBS);
            assert!(
                value
                    .limbs()
                    .get(retained..)
                    .expect("growth suffix")
                    .iter()
                    .all(|limb| *limb == 0),
                "growth initializes only newly exposed slots"
            );
            value.normalize();
            assert_eq!(value.limbs(), prefix);
        }
    }
}

#[test]
fn reused_heap_guards_preserve_fallback_state_and_commit_initialized_spans() {
    let mut inline = InternalMpUint::one();
    assert!(
        inline.try_prepare_reused_heap_limbs(2).is_none(),
        "inline storage uses the general path"
    );
    assert_eq!(inline, InternalMpUint::one());
    let mut heap = InternalMpUint::with_capacity(8);
    let pointer = heap.limbs().as_ptr();
    {
        let _abandoned = heap
            .try_prepare_reused_heap_limbs(6)
            .expect("reserved capacity");
    }
    assert_eq!(heap.limbs(), EMPTY_LIMBS);
    let mut pending = heap
        .try_prepare_reused_heap_limbs(6)
        .expect("reserved capacity");
    for index in 0..6 {
        // SAFETY: the guard reserves six aligned exclusive slots; the loop
        // writes each one exactly once before committing the logical length.
        unsafe {
            pending.as_mut_ptr().add(index).write(1);
        }
    }
    // SAFETY: every limb in the prepared six-limb span is initialized.
    assert_eq!(unsafe { pending.commit() }, &[1; 6]);
    assert_eq!(heap.limbs().as_ptr(), pointer);
    assert!(
        heap.try_prepare_reused_heap_limbs(9).is_none(),
        "insufficient capacity preserves fallback state"
    );
    assert_eq!(heap.limbs(), [1; 6]);
    assert_eq!(heap.capacity(), 8);
    assert_eq!(heap.limbs().as_ptr(), pointer);
}

#[test]
fn abandoned_writes_and_initialized_growth_preserve_existing_prefixes() {
    let check = |input: &[Limb], target: usize| {
        let mut value = InternalMpUint::from_limbs_slice(input);
        let expected = value.limbs().to_vec();
        {
            let _abandoned = value.prepare_limb_write(target);
        }
        assert_eq!(value.limbs(), expected);
        let slice = value.ensure_capacity_set_len_get_limbs(target);
        let retained = expected.len().min(target);
        assert_eq!(
            slice.get(..retained).expect("retained prefix"),
            expected.get(..retained).expect("original prefix")
        );
        assert!(
            slice
                .get(retained..)
                .expect("initialized growth suffix")
                .iter()
                .all(|limb| *limb == 0),
            "all newly exposed slots are zero"
        );
        value.normalize();
        let prefix = expected.get(..retained).expect("retained prefix");
        assert_eq!(value, InternalMpUint::from_limbs_slice(prefix));
    };
    for original in 0..=INLINE_LIMBS.checked_add(1).expect("heap boundary") {
        for target in 0..=INLINE_LIMBS.checked_add(2).expect("growth boundary") {
            check(&vec![1; original], target);
        }
    }
    let strategy = (collection::vec(any::<Limb>(), 0..=12), 0_usize..=18);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&strategy, |(limbs, target)| {
            check(&limbs, target);
            Ok(())
        })
        .expect("initialized growth property");
}

#[test]
fn carry_and_borrow_roundtrips_cross_inline_storage_boundaries() {
    for width in 0..=INLINE_LIMBS.checked_add(2).expect("heap boundary") {
        let mut value = InternalMpUint::from_limbs(vec![Limb::MAX; width]);
        value.increment();
        let mut expected = vec![0; width];
        expected.push(1);
        assert_eq!(value.limbs(), expected);
        value.decrement();
        assert_eq!(value.limbs(), vec![Limb::MAX; width]);
        let mut borrow = InternalMpUint::from_limbs(expected);
        borrow.decrement();
        assert_eq!(borrow.limbs(), vec![Limb::MAX; width]);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))
        .run(&collection::vec(any::<Limb>(), 0..=8), |limbs| {
            let mut value = InternalMpUint::from_limbs(limbs);
            let original = value.clone();
            value.increment();
            value.decrement();
            assert_eq!(value, original);
            Ok(())
        })
        .expect("unit carry property");
}

#[cfg(feature = "std")]
#[test]
fn overflowing_reservations_preserve_value_capacity_and_allocation() {
    for width in 0..=INLINE_LIMBS.checked_add(1).expect("heap boundary") {
        for reserve in [InternalMpUint::reserve, InternalMpUint::reserve_exact] {
            let mut value = InternalMpUint::from_limbs(vec![1; width]);
            let before = value.clone();
            let (capacity, pointer) = (value.capacity(), value.limbs().as_ptr());
            assert!(
                catch_unwind(AssertUnwindSafe(|| reserve(&mut value, usize::MAX))).is_err(),
                "unrepresentable reservations fail"
            );
            assert_eq!(value, before);
            assert_eq!(value.capacity(), capacity);
            assert_eq!(value.limbs().as_ptr(), pointer);
        }
    }
}
