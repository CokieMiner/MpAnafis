//! Property tests for balanced Toom-8 and unbalanced Toom-8.5.

#[cfg(feature = "std")]
use std::{sync::Barrier, thread};

#[cfg(feature = "std")]
use alloc::sync::Arc;
use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::int::logic::math::mul::{Multiplication, Schoolbook, dispatch::Widths};

use super::*;

fn balanced_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 8),
            proptest::collection::vec(any::<Limb>(), 8),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 65),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 129),
            proptest::collection::vec(any::<Limb>(), 125),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 257),
        ),
    ]
}

fn half_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 9),
            proptest::collection::vec(any::<Limb>(), 8),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 73),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 145),
            proptest::collection::vec(any::<Limb>(), 129),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 289),
        ),
    ]
}

fn square_operands() -> impl Strategy<Value = Vec<Limb>> {
    prop_oneof![
        proptest::collection::vec(any::<Limb>(), 8),
        proptest::collection::vec(any::<Limb>(), 57),
        proptest::collection::vec(any::<Limb>(), 65),
        proptest::collection::vec(any::<Limb>(), 129),
        proptest::collection::vec(any::<Limb>(), 257),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_toom8_matches_basecase((a, b) in balanced_operands()) {
        prop_assert!(Widths::new(a.len(), b.len()).toom8_balanced());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom8_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom8_half_matches_basecase((a, b) in half_operands()) {
        prop_assert!(!Widths::new(a.len(), b.len()).toom8_balanced());
        prop_assert!(Widths::new(a.len(), b.len()).toom8_half_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom8_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom8_square_matches_basecase(a in square_operands()) {
        prop_assert!(Multiplication::operand_has_eight_parts(a.len()));
        let result_len = a.len().wrapping_mul(2);
        let mut expected = vec![0; result_len];
        let mut actual = vec![0; result_len];
        Schoolbook::sqr(&mut expected, &a);

        let scratch_len = Multiplication::toom8_sqr_scratch_len(a.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom8::sqr(&mut actual, &a, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}

#[cfg(feature = "std")]
#[test]
fn concurrent_scratch_prefix_growth_is_stable() {
    const WIDTHS: [usize; 8] = [8192, 12_288, 16_384, 20_480, 24_576, 28_672, 32_768, 36_864];

    let start = Arc::new(Barrier::new(WIDTHS.len()));
    let handles: Vec<_> = WIDTHS
        .into_iter()
        .map(|width| {
            let worker_start = Arc::clone(&start);
            thread::spawn(move || {
                let _ = worker_start.wait();
                let mul = Multiplication::toom8_mul_scratch_len(width, width);
                let square = Multiplication::toom8_sqr_scratch_len(width);
                assert!(
                    mul != 0 && square != 0,
                    "wide Toom-8 scratch must be nonempty"
                );
                assert_eq!(
                    mul,
                    Multiplication::toom8_mul_scratch_len(width, width),
                    "cached multiplication prefix changed after concurrent growth"
                );
                assert_eq!(
                    square,
                    Multiplication::toom8_sqr_scratch_len(width),
                    "cached square prefix changed after concurrent growth"
                );
            })
        })
        .collect();

    for handle in handles {
        assert!(
            handle.join().is_ok(),
            "concurrent Toom-8 prefix worker panicked"
        );
    }
}
