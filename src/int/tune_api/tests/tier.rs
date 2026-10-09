//! Production multiplication and squaring calls with retained scratch.

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::vec;

use proptest::prelude::{ProptestConfig, proptest};

use crate::tune_api::{Limb, MultiplicationBenchState, SquaringBenchState};

use super::strategies::{integer, limbs};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn dispatchers_reuse_scratch_and_preserve_surplus_destination_limbs(
        left in limbs(if cfg!(miri) { 8 } else { 128 }),
        right in limbs(if cfg!(miri) { 8 } else { 128 }),
    ) {
        let mut multiplication = MultiplicationBenchState::default();
        let mut squaring = SquaringBenchState::default();
        for (a, b) in [(&left, &right), (&left, &left), (&right, &left)] {
            let expected = integer(a).checked_mul(&integer(b)).expect("unlimited product");
            let square_value = integer(a);
            let square = square_value.checked_mul(&square_value).expect("unlimited square");
            let product_width = a.len().checked_add(b.len()).expect("small test width");
            let square_width = a.len().checked_mul(2).expect("small test width");
            for surplus in [0, 3] {
                let product_len = product_width.checked_add(surplus).expect("small test width");
                let square_len = square_width.checked_add(surplus).expect("small test width");
                let mut product_output = vec![Limb::MAX; product_len];
                let mut square_output = vec![Limb::MAX; square_len];
                for poison in [Limb::MAX, 7] {
                    product_output.fill(poison);
                    square_output.fill(poison);
                    {
                        let mut product_call = multiplication.prepare(&mut product_output, a, b);
                        let mut square_call = squaring.prepare(&mut square_output, a);
                        product_call.run();
                        product_call.run();
                        square_call.run();
                        square_call.run();
                    }
                    let (product_span, product_suffix) = product_output.split_at(product_width);
                    let (square_span, square_suffix) = square_output.split_at(square_width);
                    assert_eq!(integer(product_span), expected);
                    assert_eq!(integer(square_span), square);
                    assert!(product_suffix.iter().all(|word| *word == poison));
                    assert!(square_suffix.iter().all(|word| *word == poison));
                }
            }
        }
    }
}

#[test]
fn preparation_rejects_empty_inputs_and_short_destinations_without_writes() {
    let mut multiplication = MultiplicationBenchState::default();
    let mut squaring = SquaringBenchState::default();
    for (a, b, length) in [
        (&[][..], &[1][..], 1),
        (&[1][..], &[][..], 1),
        (&[1][..], &[2][..], 1),
    ] {
        let mut output = vec![7; length];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _prepared = multiplication.prepare(&mut output, a, b);
            }))
            .is_err()
        );
        assert!(output.iter().all(|word| *word == 7));
    }
    for (a, length) in [(&[][..], 1), (&[1][..], 1)] {
        let mut output = vec![7; length];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _prepared = squaring.prepare(&mut output, a);
            }))
            .is_err()
        );
        assert!(output.iter().all(|word| *word == 7));
    }
}
