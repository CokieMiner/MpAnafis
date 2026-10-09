//! Normalized division inputs, remainder policies, and reusable output buffers.

use core::mem::swap;
use std::panic::catch_unwind;

use proptest::prelude::{ProptestConfig, prop_assert_eq, proptest};

use crate::tune_api::{DivisionAlgorithm, DivisionRunner};

use super::strategies::{integer, limbs};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]

    #[test]
    fn algorithms_preserve_quotients_and_remainders_across_reuse(
        mut numerator_words in limbs(8), mut denominator_words in limbs(8),
    ) {
        if denominator_words.iter().all(|word| *word == 0) {
            *denominator_words.first_mut().expect("generated limbs are nonempty") = 1;
        }
        if integer(&numerator_words) < integer(&denominator_words) {
            swap(&mut numerator_words, &mut denominator_words);
        }
        // Swapping can move a zero numerator into the denominator.
        if denominator_words.iter().all(|word| *word == 0) {
            *denominator_words.first_mut().expect("generated limbs are nonempty") = 1;
        }
        numerator_words.extend_from_slice(&[0, 0]);
        denominator_words.extend_from_slice(&[0, 0]);
        let numerator = integer(&numerator_words);
        let denominator = integer(&denominator_words);
        let (quotient, remainder) = numerator.div_rem(&denominator).expect("nonzero divisor");
        let mut runner = DivisionRunner::new(&numerator_words, &denominator_words);
        for _ in 0..2 {
            for algorithm in [
                DivisionAlgorithm::Production, DivisionAlgorithm::AlgorithmD,
                DivisionAlgorithm::BurnikelZiegler, DivisionAlgorithm::NewtonRaphson,
            ] {
                runner.run::<true>(algorithm);
                prop_assert_eq!(&integer(runner.quotient_limbs()), &quotient);
                prop_assert_eq!(&integer(runner.remainder_limbs()), &remainder);
            }
            for algorithm in [
                DivisionAlgorithm::Production, DivisionAlgorithm::AlgorithmD,
                DivisionAlgorithm::BurnikelZiegler, DivisionAlgorithm::NewtonRaphson,
            ] {
                runner.run::<false>(algorithm);
                prop_assert_eq!(&integer(runner.quotient_limbs()), &quotient);
            }
        }
    }
}

#[test]
fn construction_rejects_zero_divisors_and_smaller_numerators() {
    for (numerator, denominator) in [
        (&[1][..], &[][..]),
        (&[1][..], &[0, 0][..]),
        (&[][..], &[1][..]),
        (&[2, 0][..], &[3, 0][..]),
    ] {
        assert!(catch_unwind(|| DivisionRunner::new(numerator, denominator)).is_err());
    }
}
