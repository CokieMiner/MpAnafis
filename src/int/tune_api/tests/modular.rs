//! Odd-modulus exponentiation, canonical results, and runner reuse.

use std::panic::catch_unwind;

use proptest::test_runner::{Config, TestRunner};

use crate::{
    MpUint,
    tune_api::{ModularPowAlgorithm, ModularPowRunner},
};

use super::strategies::{integer, limbs};

#[test]
fn algorithms_return_owned_residues_across_reuse() {
    for (base, exponent, modulus) in [(3, 5, 7), (0, 9, 7), (5, 0, 7), (2, 4, 1), (0, 0, 1)] {
        check_runner(&[base], exponent, &[modulus]);
    }
    let mut cases = TestRunner::new(Config {
        cases: if cfg!(miri) { 4 } else { 32 },
        source_file: Some(file!()),
        ..Config::default()
    });
    cases
        .run(
            &(limbs(5), 0_usize..=32, limbs(5)),
            |(base, exponent, mut modulus)| {
                *modulus.first_mut().expect("generated limbs are nonempty") |= 1;
                check_runner(&base, exponent, &modulus);
                Ok(())
            },
        )
        .expect("modular runner contracts hold for generated operands");
}

#[test]
fn construction_rejects_zero_and_even_moduli() {
    for modulus in [&[][..], &[0, 0][..], &[2][..], &[0, 1][..]] {
        assert!(catch_unwind(|| ModularPowRunner::new(&[3], &[2], modulus)).is_err());
    }
}

fn check_runner(base: &[usize], exponent: usize, modulus: &[usize]) {
    let expected = integer(base)
        .pow_mod(&MpUint::from(exponent), &integer(modulus))
        .expect("nonzero modulus");
    let mut runner = ModularPowRunner::new(base, &[exponent, 0], modulus);
    let retained = runner.run(ModularPowAlgorithm::Production);
    for _ in 0..2 {
        for algorithm in [
            ModularPowAlgorithm::Production,
            ModularPowAlgorithm::Montgomery,
            ModularPowAlgorithm::Barrett,
        ] {
            let result = runner.run(algorithm);
            assert_eq!(result, retained);
            assert_eq!(integer(result.as_ref()), expected);
            assert!(result.as_ref().last().is_none_or(|word| *word != 0));
        }
    }
    drop(runner);
    assert_eq!(integer(retained.as_ref()), expected);
}
