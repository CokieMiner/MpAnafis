//! GCD algorithm selection, input normalization, and result ownership.

use proptest::test_runner::{Config, TestRunner};

use crate::tune_api::{
    GcdAlgorithm, GcdRunner, LehmerSimAlgorithm, LehmerSimRunner, LehmerUpdateAlgorithm,
    LehmerUpdateRunner,
};

use super::strategies::{integer, limbs};

#[test]
fn algorithms_return_owned_canonical_results_across_reuse() {
    for (left, right) in [(0, 0), (0, 42), (48, 18), (35, 64)] {
        check_runners(&[left], &[right]);
    }
    let max_len = if cfg!(miri) { 8 } else { 64 };
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(&(limbs(max_len), limbs(max_len)), |(left, right)| {
            check_runners(&left, &right);
            Ok(())
        })
        .expect("GCD runner contracts hold for generated operands");
}

fn check_runners(left: &[usize], right: &[usize]) {
    let expected = integer(left).gcd(&integer(right));
    let mut left_words = left.to_vec();
    let mut right_words = right.to_vec();
    left_words.extend_from_slice(&[0, 0]);
    right_words.extend_from_slice(&[0, 0]);
    let mut runner = GcdRunner::new(&left_words, &right_words);
    let retained = runner.run(GcdAlgorithm::Production);
    for _ in 0..2 {
        for algorithm in [
            GcdAlgorithm::Production,
            GcdAlgorithm::Lehmer,
            GcdAlgorithm::HalfGcd,
        ] {
            let result = runner.run(algorithm);
            assert_eq!(result, retained);
            assert_eq!(integer(result.as_ref()), expected);
            assert!(result.as_ref().last().is_none_or(|word| *word != 0));
        }
    }
    drop(runner);
    assert_eq!(integer(retained.as_ref()), expected);
    let mut simulation = LehmerSimRunner::new(&left_words, &right_words);
    for algorithm in [LehmerSimAlgorithm::Narrow, LehmerSimAlgorithm::Wide] {
        assert_eq!(integer(simulation.run(algorithm).as_ref()), expected);
    }
    let mut update = LehmerUpdateRunner::new(&left_words, &right_words);
    for algorithm in [
        LehmerUpdateAlgorithm::Fused,
        LehmerUpdateAlgorithm::Separate,
    ] {
        assert_eq!(integer(update.run(algorithm).as_ref()), expected);
    }
}
