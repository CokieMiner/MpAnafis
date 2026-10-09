//! Formatting tier agreement and recursive cache reuse.

use std::panic::catch_unwind;

use crate::{
    MpUint,
    tune_api::{FormattingAlgorithm, FormattingRunner, Limb},
};

#[test]
fn algorithms_produce_equal_owned_output_across_cache_reuse() {
    let widths: &[usize] = if cfg!(miri) {
        &[1, 5]
    } else {
        &[1, 4, 5, 16, 17, 32, 33, 64]
    };
    for radix in 3_u32..=36 {
        if radix.is_power_of_two() || (cfg!(miri) && ![3, 10, 36].contains(&radix)) {
            continue;
        }
        for &width in widths {
            let mut schoolbook =
                FormattingRunner::new(FormattingAlgorithm::Schoolbook, width, radix);
            let expected = schoolbook.output();
            let value = MpUint::from_str_radix(&expected, radix).expect("valid formatted digits");
            let width_bits = width
                .checked_mul(usize::try_from(Limb::BITS).expect("limb bits fit usize"))
                .expect("small test width");
            assert!(value.significant_bits() <= width_bits);
            assert!(
                value.significant_bits()
                    > width_bits
                        .checked_sub(usize::try_from(Limb::BITS).expect("limb bits fit usize"))
                        .expect("nonempty operand")
            );
            for algorithm in [
                FormattingAlgorithm::Schoolbook,
                FormattingAlgorithm::Recursive,
            ] {
                let mut runner = FormattingRunner::new(algorithm, width, radix);
                let retained = runner.output();
                for _ in 0..2 {
                    runner.run();
                    assert_eq!(
                        runner.output(),
                        expected,
                        "{algorithm:?}, width {width}, radix {radix}"
                    );
                }
                drop(runner);
                assert_eq!(retained, expected);
            }
        }
    }
}

#[test]
fn construction_rejects_empty_widths_and_unsupported_radices() {
    for algorithm in [
        FormattingAlgorithm::Schoolbook,
        FormattingAlgorithm::Recursive,
    ] {
        assert!(catch_unwind(|| FormattingRunner::new(algorithm, 0, 10)).is_err());
        for radix in [0, 1, 2, 4, 8, 16, 32, 37, u32::MAX] {
            assert!(catch_unwind(|| FormattingRunner::new(algorithm, 1, radix)).is_err());
        }
    }
}
