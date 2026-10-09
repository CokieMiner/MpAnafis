//! Complete first writes into uninitialized products and untouched suffixes.

#![expect(
    unsafe_code,
    reason = "Exact nonempty spans and sized scratch satisfy product writers; sentinels are initialized before inspection"
)]

use core::mem::MaybeUninit;

use alloc::vec;

use proptest::prelude::*;

use crate::int::logic::unsigned::math::mul::{
    Limb, LimbOutput, MulPlan, MulScratch, Multiplication, Schoolbook,
};

#[test]
fn forced_tiers_initialize_complete_products() {
    for (plan, a_len, b_len) in [
        (MulPlan::Schoolbook, 5_usize, 4_usize),
        (MulPlan::Karatsuba, 4, 4),
        (MulPlan::Toom3, 6, 6),
        (MulPlan::Toom32, 7, 5),
        (MulPlan::Toom43, 10, 7),
        (MulPlan::Toom4, 8, 8),
        (MulPlan::Lopsided, 15, 5),
        (MulPlan::Karatsuba, 32, 32),
        (MulPlan::Toom3, 33, 33),
        (MulPlan::Toom32, 48, 32),
        (MulPlan::Toom43, 64, 48),
        (MulPlan::Toom4, 48, 48),
        (MulPlan::Toom6, 6, 6),
        (MulPlan::Toom6, 11, 11),
        (MulPlan::Toom6, 48, 48),
        (MulPlan::Toom6, 49, 42),
        (MulPlan::Toom8, 8, 8),
        (MulPlan::Toom8, 15, 15),
        (MulPlan::Toom8, 17, 16),
        (MulPlan::Toom8, 64, 64),
        (MulPlan::Toom8, 65, 64),
        (MulPlan::Lopsided, 96, 32),
    ] {
        if cfg!(miri) && a_len.max(b_len) > 17 {
            continue;
        }
        for (left_word, right_word) in [(0, Limb::MAX), (Limb::MAX, Limb::MAX), (1, 3)] {
            let a = vec![left_word; a_len];
            let b = vec![right_word; b_len];
            let mut expected = vec![0; a_len.checked_add(b_len).expect("product fits")];
            Schoolbook::mul_nonempty_distinct(&mut expected, &a, &b);
            let mut output = vec![MaybeUninit::uninit(); expected.len()];
            let mut scratch = vec![Limb::MAX; Multiplication::scratch_len(plan, a_len, b_len)];
            Multiplication::execute_plan(plan, &mut output, &a, &b, &mut scratch);
            // SAFETY: the selected kernel initializes every exact product limb.
            // Miri checks actual uninitialized storage before this read.
            let initialized = unsafe { LimbOutput::assume_init(&output) };
            assert_eq!(initialized, expected, "{plan:?} {a_len}x{b_len}");
        }
    }
}

#[test]
fn uninitialized_products_preserve_surplus_and_reuse_dirty_arena() {
    let mut scratch = MulScratch::default();
    scratch.prepare(16_384);
    let arena_pointer = scratch.buf.as_ptr();
    for (a_len, b_len) in [
        (1_usize, 1_usize),
        (4, 4),
        (5, 5),
        (20, 20),
        (23, 23),
        (24, 24),
        (32, 32),
        (48, 48),
        (64, 48),
        (97, 71),
        (144, 144),
    ] {
        if cfg!(miri) && a_len.max(b_len) > 32 {
            continue;
        }
        let a = vec![Limb::MAX; a_len];
        let b = vec![Limb::MAX; b_len];
        let width = a_len.checked_add(b_len).expect("product fits");
        let mut expected = vec![0; width];
        Schoolbook::mul_nonempty_distinct(&mut expected, &a, &b);
        let mut output = vec![MaybeUninit::uninit(); width.checked_add(2).expect("sentinels fit")];
        let (_, suffix) = output.split_at_mut(width);
        suffix.copy_from_slice(&[MaybeUninit::new(17), MaybeUninit::new(29)]);
        scratch.buf.fill(Limb::MAX);
        // SAFETY: separate nonempty operands, the complete destination, and
        // sized scratch satisfy the initializer's disjointness and width contract.
        let initialized = unsafe {
            Multiplication::mul_nonempty_distinct_into_uninit(&a, &b, &mut output, &mut scratch)
        };
        assert_eq!(initialized, expected, "uninitialized {a_len}x{b_len}");
        // SAFETY: both suffix sentinels were initialized before the call.
        let guards = unsafe { LimbOutput::assume_init(output.get(width..).expect("suffix fits")) };
        assert_eq!(guards, &[17, 29]);
        assert_eq!(scratch.buf.as_ptr(), arena_pointer);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 24 }))]

    #[test]
    fn arbitrary_uninitialized_products_match_schoolbook(
        a in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 129 }),
        b in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 8 } else { 129 }),
    ) {
        let width = a.len().checked_add(b.len()).expect("product fits");
        let mut expected = vec![0; width];
        Schoolbook::mul_nonempty_distinct(&mut expected, &a, &b);
        let mut output = vec![MaybeUninit::uninit(); width];
        let mut scratch = MulScratch::default();
        scratch.prepare(Multiplication::required_scratch(a.len(), b.len()));
        scratch.buf.fill(Limb::MAX);
        // SAFETY: independent nonempty operands and sized writable storage
        // satisfy the initializer's complete-product and scratch contracts.
        let initialized = unsafe {
            Multiplication::mul_nonempty_distinct_into_uninit(&a, &b, &mut output, &mut scratch)
        };
        prop_assert_eq!(initialized, expected);
    }
}
