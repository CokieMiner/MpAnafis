//! Forced squaring tiers and complete overwrite during prepared execution.

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::vec;

use proptest::test_runner::{Config, TestRunner};

use crate::tune_api::{Limb, SquaringAlgorithm, SquaringRunner};

use super::strategies::{integer, limbs};

#[test]
fn forced_tiers_overwrite_dirty_outputs_across_prepared_reuse() {
    let algorithms = [
        SquaringAlgorithm::Schoolbook,
        SquaringAlgorithm::Karatsuba,
        SquaringAlgorithm::ToomCook3,
        SquaringAlgorithm::ToomCook4,
        SquaringAlgorithm::ToomCook6,
        SquaringAlgorithm::ToomCook85,
    ];
    let widths: &[usize] = if cfg!(miri) {
        &[1, 4, 5]
    } else {
        &[1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128, 256]
    };
    for &width in widths {
        for fill in [0, 1, Limb::MAX] {
            let input = vec![fill; width];
            for algorithm in algorithms {
                check_runner(algorithm, &input);
            }
        }
    }
    let max_len = if cfg!(miri) { 8 } else { 256 };
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(&limbs(max_len), |input| {
            for algorithm in algorithms {
                check_runner(algorithm, &input);
            }
            Ok(())
        })
        .expect("squaring runner contracts hold for generated operands");
}

#[cfg(not(target_pointer_width = "16"))]
#[test]
#[cfg_attr(
    miri,
    ignore = "The full SSA strategy matrix at 64..256 limbs is prohibitively slow under Miri"
)]
fn ssa_strategies_overwrite_dense_and_zero_padded_outputs() {
    for algorithm in [
        SquaringAlgorithm::SsaForced,
        SquaringAlgorithm::SsaProduction,
        SquaringAlgorithm::SsaDirectFermat,
    ] {
        for width in [64_usize, 128, 256] {
            for significant in [0, 1, width] {
                let mut input = vec![0; width];
                input
                    .get_mut(..significant)
                    .expect("prefix fits the operand")
                    .fill(Limb::MAX);
                check_runner(algorithm, &input);
            }
        }
    }
}

#[test]
fn validation_rejects_invalid_widths_before_modifying_output() {
    for width in [0, usize::MAX] {
        assert!(
            catch_unwind(|| SquaringRunner::new(SquaringAlgorithm::Schoolbook, width)).is_err()
        );
    }
    let mut runner = SquaringRunner::new(SquaringAlgorithm::Schoolbook, 2);
    for (input, output_len) in [(&[1][..], 4), (&[1, 2][..], 3), (&[1, 2][..], 5)] {
        let mut output = vec![7; output_len];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _prepared = runner.prepare(&mut output, input);
            }))
            .is_err()
        );
        assert!(output.iter().all(|word| *word == 7));
    }
}

fn check_runner(algorithm: SquaringAlgorithm, input: &[Limb]) {
    let value = integer(input);
    let expected = value
        .checked_mul(&value)
        .expect("unlimited reference square");
    let output_len = input.len().checked_mul(2).expect("small test width");
    let mut runner = SquaringRunner::new(algorithm, input.len());
    let mut output = vec![0; output_len];
    for poison in [Limb::MAX, 7] {
        output.fill(poison);
        {
            let mut prepared = runner.prepare(&mut output, input);
            prepared.run();
            prepared.run();
        }
        assert_eq!(
            integer(&output),
            expected,
            "{algorithm:?}, width {}",
            input.len()
        );
    }
    output.fill(Limb::MAX);
    runner.run(&mut output, input);
    assert_eq!(integer(&output), expected);
}
