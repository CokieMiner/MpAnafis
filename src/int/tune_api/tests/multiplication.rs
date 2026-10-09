//! Forced multiplication tiers, shared inputs, and complete destination overwrite.

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::vec;

use proptest::test_runner::{Config, TestRunner};

use crate::tune_api::{Limb, MultiplicationAlgorithm, MultiplicationRunner};

use super::strategies::{integer, limbs};

#[test]
fn forced_tiers_overwrite_dirty_outputs_across_prepared_reuse() {
    let algorithms = [
        MultiplicationAlgorithm::Schoolbook,
        MultiplicationAlgorithm::Karatsuba,
        MultiplicationAlgorithm::ToomCook3,
        MultiplicationAlgorithm::ToomCook4,
        MultiplicationAlgorithm::ToomCook6,
        MultiplicationAlgorithm::ToomCook85,
    ];
    let shapes: &[(usize, usize)] = if cfg!(miri) {
        &[(1, 1), (4, 5)]
    } else {
        &[
            (1, 1),
            (2, 3),
            (4, 5),
            (7, 8),
            (15, 16),
            (17, 31),
            (32, 33),
            (64, 128),
        ]
    };
    for &(left_width, right_width) in shapes {
        for fill in [0, 1, Limb::MAX] {
            let left = vec![fill; left_width];
            let right = vec![fill; right_width];
            for algorithm in algorithms {
                check_runner(algorithm, &left, &right);
            }
        }
    }
    let max_len = if cfg!(miri) { 8 } else { 128 };
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(&(limbs(max_len), limbs(max_len)), |(left, right)| {
            for algorithm in algorithms {
                check_runner(algorithm, &left, &right);
                check_runner(algorithm, &left, &left);
            }
            Ok(())
        })
        .expect("multiplication runner contracts hold for generated operands");
}

#[cfg(not(target_pointer_width = "16"))]
#[test]
#[cfg_attr(
    miri,
    ignore = "The full SSA strategy matrix at 64..256 limbs is prohibitively slow under Miri"
)]
fn ssa_strategies_overwrite_dense_and_zero_padded_rectangular_outputs() {
    for algorithm in [
        MultiplicationAlgorithm::SsaForced,
        MultiplicationAlgorithm::SsaProduction,
        MultiplicationAlgorithm::SsaCrt,
        MultiplicationAlgorithm::SsaDirectFermat,
    ] {
        for (left_width, right_width) in [
            (64_usize, 64_usize),
            (65, 63),
            (128, 64),
            (128, 129),
            (256, 256),
        ] {
            for significant in [0, 1, left_width] {
                let mut left = vec![0; left_width];
                left.get_mut(..significant)
                    .expect("prefix fits the operand")
                    .fill(Limb::MAX);
                let right = vec![Limb::MAX; right_width];
                check_runner(algorithm, &left, &right);
            }
        }
    }
}

#[test]
fn validation_rejects_invalid_widths_before_modifying_output() {
    for (left_width, right_width) in [(0, 1), (1, 0), (usize::MAX, 1)] {
        assert!(
            catch_unwind(|| MultiplicationRunner::new(
                MultiplicationAlgorithm::Schoolbook,
                left_width,
                right_width
            ))
            .is_err()
        );
    }
    let mut runner = MultiplicationRunner::new(MultiplicationAlgorithm::Schoolbook, 2, 3);
    for (left, right, output_len) in [
        (&[1][..], &[1, 2, 3][..], 5),
        (&[1, 2][..], &[1, 2][..], 5),
        (&[1, 2][..], &[1, 2, 3][..], 4),
        (&[1, 2][..], &[1, 2, 3][..], 6),
    ] {
        let mut output = vec![7; output_len];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _prepared = runner.prepare(&mut output, left, right);
            }))
            .is_err()
        );
        assert!(output.iter().all(|word| *word == 7));
    }
}

fn check_runner(algorithm: MultiplicationAlgorithm, left: &[Limb], right: &[Limb]) {
    let expected = integer(left)
        .checked_mul(&integer(right))
        .expect("unlimited reference product");
    let output_len = left
        .len()
        .checked_add(right.len())
        .expect("small test width");
    let mut runner = MultiplicationRunner::new(algorithm, left.len(), right.len());
    let mut output = vec![0; output_len];
    for poison in [Limb::MAX, 7] {
        output.fill(poison);
        {
            let mut prepared = runner.prepare(&mut output, left, right);
            prepared.run();
            prepared.run();
        }
        assert_eq!(
            integer(&output),
            expected,
            "{algorithm:?}, widths {} and {}",
            left.len(),
            right.len()
        );
    }
    output.fill(Limb::MAX);
    runner.run(&mut output, left, right);
    assert_eq!(integer(&output), expected);
}
