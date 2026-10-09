//! Complete parsing calls, deterministic chunk widths, and owned results.

use std::panic::catch_unwind;

use crate::{
    MpUint,
    tune_api::{Limb, ParsingAlgorithm, ParsingRunner},
};

use super::strategies::integer;

#[test]
fn algorithms_parse_equal_full_chunks_and_return_owned_results() {
    let chunks: &[usize] = if cfg!(miri) {
        &[1, 2, 17]
    } else {
        &[1, 2, 7, 16, 17, 18, 19, 20, 64, 128]
    };
    for radix in 3_u32..=36 {
        if radix.is_power_of_two() || (cfg!(miri) && ![3, 10, 36].contains(&radix)) {
            continue;
        }
        let native_radix = Limb::try_from(radix).expect("supported radices fit every limb width");
        let mut power = native_radix;
        let mut chunk_digits = 1_usize;
        while let Some(next) = power.checked_mul(native_radix) {
            power = next;
            chunk_digits = chunk_digits
                .checked_add(1)
                .expect("native chunk width fits usize");
        }
        for &chunk_count in chunks {
            let schoolbook = ParsingRunner::new(ParsingAlgorithm::Schoolbook, chunk_count, radix);
            let expected = schoolbook.run();
            let digit_count = chunk_count
                .checked_mul(chunk_digits)
                .expect("small test input");
            assert_eq!(schoolbook.input().len(), digit_count);
            assert!(!schoolbook.input().starts_with('0'));
            let reference =
                MpUint::from_str_radix(schoolbook.input(), radix).expect("valid prepared input");
            assert_eq!(integer(expected.as_ref()), reference);
            for algorithm in [
                ParsingAlgorithm::Schoolbook,
                ParsingAlgorithm::Recursive,
                ParsingAlgorithm::Production,
            ] {
                let runner = ParsingRunner::new(algorithm, chunk_count, radix);
                assert_eq!(runner.input(), schoolbook.input());
                assert!(runner.verify());
                let retained = runner.run();
                assert_eq!(runner.run(), expected);
                drop(runner);
                assert_eq!(retained, expected);
            }
        }
    }
}

#[test]
fn construction_rejects_invalid_radices_and_unrepresentable_input_lengths() {
    for algorithm in [
        ParsingAlgorithm::Schoolbook,
        ParsingAlgorithm::Recursive,
        ParsingAlgorithm::Production,
    ] {
        for chunks in [0, usize::MAX] {
            assert!(catch_unwind(|| ParsingRunner::new(algorithm, chunks, 10)).is_err());
        }
        for radix in [0, 1, 2, 4, 8, 16, 32, 37, u32::MAX] {
            assert!(catch_unwind(|| ParsingRunner::new(algorithm, 1, radix)).is_err());
        }
    }
}
