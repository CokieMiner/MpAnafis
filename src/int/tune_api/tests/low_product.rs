//! Forced low products, prepared reuse, shared inputs, and validation boundaries.

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use alloc::{vec, vec::Vec};

use proptest::test_runner::{Config, TestRunner};

use crate::tune_api::{Limb, LowProductAlgorithm, LowProductRunner};

use super::strategies::{integer, limbs};

#[test]
fn forced_roots_preserve_prefixes_and_suffixes_across_prepared_reuse() {
    let widths: &[usize] = if cfg!(miri) {
        &[1, 2, 3, 4, 5, 8]
    } else {
        &[
            1, 2, 3, 4, 5, 8, 16, 32, 48, 64, 71, 72, 73, 128, 258, 259, 260, 512,
        ]
    };
    for &len in widths {
        let padded_len = len.checked_add(1).expect("test input guard");
        for fill in [0, 1, Limb::MAX] {
            let left = vec![fill; padded_len];
            let right: Vec<Limb> = left.iter().map(|word| word.rotate_left(1)).collect();
            check_runners(len, &left, &right);
            check_runners(len, &left, &left);
            check_runners(
                len,
                &left,
                left.get(1..).expect("overlapping operand prefix"),
            );
        }
    }
    let max_len = if cfg!(miri) { 8 } else { 512 };
    TestRunner::new(Config {
        cases: if cfg!(miri) { 2 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    })
    .run(&(limbs(max_len), limbs(max_len)), |(left, right)| {
        let len = left.len().min(right.len());
        check_runners(len, &left, &right);
        check_runners(len, &right, &left);
        check_runners(len, &left, &left);
        Ok(())
    })
    .expect("low-product runner contracts hold for generated operands");
}

#[test]
fn invalid_widths_are_rejected_before_output_changes() {
    for algorithm in [
        LowProductAlgorithm::Schoolbook,
        LowProductAlgorithm::Mulders,
        LowProductAlgorithm::Full,
    ] {
        assert!(catch_unwind(|| LowProductRunner::new(algorithm, 0)).is_err());
        let mut runner = LowProductRunner::new(algorithm, 4);
        for (left, right, output_len) in [
            (&[1, 2, 3][..], &[1, 2, 3, 4][..], 4),
            (&[1, 2, 3, 4][..], &[1, 2, 3][..], 4),
            (&[1, 2, 3, 4][..], &[1, 2, 3, 4][..], 3),
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
        let mut recovered = [Limb::MAX; 4];
        runner.run(&mut recovered, &[1, 2, 3, 4], &[1, 0, 0, 0]);
        assert_eq!(recovered, [1, 2, 3, 4]);
    }
    for algorithm in [LowProductAlgorithm::Mulders, LowProductAlgorithm::Full] {
        assert!(catch_unwind(|| LowProductRunner::new(algorithm, usize::MAX)).is_err());
    }
}

fn check_runners(len: usize, left: &[Limb], right: &[Limb]) {
    let limb_bits = usize::try_from(Limb::BITS).expect("native limb width fits usize");
    let bits = len.checked_mul(limb_bits).expect("bounded bit width");
    let expected = integer(left.get(..len).expect("left prefix"))
        .checked_mul(&integer(right.get(..len).expect("right prefix")))
        .expect("unlimited reference product")
        .bit_range(0, bits);
    for algorithm in [
        LowProductAlgorithm::Schoolbook,
        LowProductAlgorithm::Mulders,
        LowProductAlgorithm::Full,
    ] {
        let mut runner = LowProductRunner::new(algorithm, len);
        let mut output = vec![0; len.checked_add(2).expect("two output sentinels")];
        for poison in [Limb::MAX, 7] {
            output.fill(poison);
            {
                let (_, destination) = output.split_at_mut(1);
                let mut prepared = runner.prepare(destination, left, right);
                prepared.run();
                prepared.run();
            }
            let (_, active) = output.split_at(1);
            assert_eq!(integer(active.get(..len).expect("output prefix")), expected);
            assert_eq!(
                (output.first(), output.last()),
                (Some(&poison), Some(&poison))
            );
        }
        output.fill(29);
        let (_, destination) = output.split_at_mut(1);
        runner.run(destination, left, right);
        assert_eq!(
            integer(destination.get(..len).expect("output prefix")),
            expected
        );
        assert_eq!((output.first(), output.last()), (Some(&29), Some(&29)));
    }
}
