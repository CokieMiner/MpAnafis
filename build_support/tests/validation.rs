//! Finite inputs and threshold-chain predicates.

use super::super::Validation;

#[test]
fn finite_and_optional_inputs_obey_sentinel_and_predecessor_bounds() {
    let values = [0, 1, 2, 17, usize::MAX - 2, usize::MAX - 1, usize::MAX];
    for value in values {
        assert_eq!(
            Validation::valid_finite(value),
            (1..usize::MAX - 1).contains(&value)
        );
        for predecessor in values {
            let expected = value == 0 || (value < usize::MAX - 1 && value > predecessor);
            assert_eq!(
                Validation::valid_optional_crossover(value, predecessor),
                expected,
                "{value} after {predecessor}"
            );
        }
    }
}

#[test]
fn threshold_chains_allow_equal_entries_and_only_disabled_tails() {
    let values = [0, 1, 2, usize::MAX - 2, usize::MAX - 1, usize::MAX];
    assert!(
        Validation::valid_threshold_chain(&[]),
        "empty chain has no constraints"
    );
    for first in values {
        for second in values {
            for third in values {
                let chain = [first, second, third];
                let expected = chain.iter().all(|&entry| entry != 0 && entry != usize::MAX)
                    && first <= second
                    && second <= third;
                assert_eq!(
                    Validation::valid_threshold_chain(&chain),
                    expected,
                    "{chain:?}"
                );
            }
        }
    }
}

#[test]
fn transforms_follow_the_last_active_tier_or_are_disabled() {
    for chain in [
        &[][..],
        &[usize::MAX - 1][..],
        &[1, 2, 3][..],
        &[1, 2, usize::MAX - 1][..],
        &[1, usize::MAX - 1, usize::MAX - 1][..],
    ] {
        let predecessor = chain
            .iter()
            .copied()
            .filter(|&value| value != usize::MAX - 1)
            .max()
            .unwrap_or(0);
        let top = chain.last().copied().unwrap_or(usize::MAX - 1);
        for value in [0, 1, 2, 3, 4, usize::MAX - 2, usize::MAX - 1, usize::MAX] {
            let expected = value == 0 || (value > predecessor && value < usize::MAX - 1);
            assert_eq!(
                Validation::valid_transform_crossover(value, top, chain),
                expected,
                "{value} after {chain:?}"
            );
        }
    }
}
