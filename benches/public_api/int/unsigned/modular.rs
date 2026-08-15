//! Modular arithmetic: `add_mod`, `sub_mod`, `mul_mod`, `pow_mod`, `invert`,
//! `montgomery_mul`, `barrett_reduce`.
//!
//! Every modulus is odd, which is both the realistic shape and the precondition
//! the Montgomery domain requires.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use core::ops::{Add, Mul, Rem, Sub};

use divan::black_box;
use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::{Integer, ops::RemRounding};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
use crate::int::{
    ladders::{EXTENDED_GCD, MODULAR, MODULAR_EXP},
    support::{
        SAMPLE_COUNT_HEAVY, SAMPLE_SIZE_FAST, SAMPLE_SIZE_HEAVY, coprime_hex_pair, mp_uint, odd_hex,
    },
};

/// `(a + b) mod m`.
///
/// GMP has no fused modular addition; the counterpart is the add-then-reduce a
/// caller would write.
mod add_mod {
    use super::*;

    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_uint(bits, 42);
        let right = mp_uint(bits, 1_337);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&left).add_mod(black_box(&right), black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_uint(bits, 42);
        let right = rug_uint(bits, 1_337);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let sum = Integer::from(Add::add(black_box(&left), black_box(&right)));
            let _output = black_box(Integer::from(Rem::rem(&sum, black_box(&modulus))));
        });
    }
}

/// `(a - b) mod m`.
mod sub_mod {
    use super::*;

    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_uint(bits, 42);
        let right = mp_uint(bits, 1_337);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&left).sub_mod(black_box(&right), black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_uint(bits, 42);
        let right = rug_uint(bits, 1_337);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let difference = Integer::from(Sub::sub(black_box(&left), black_box(&right)));
            let _output = black_box(Integer::from(RemRounding::rem_euc(
                &difference,
                black_box(&modulus),
            )));
        });
    }
}

/// `(a * b) mod m`.
mod mul_mod {
    use super::*;

    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_uint(bits, 42);
        let right = mp_uint(bits, 1_337);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&left).mul_mod(black_box(&right), black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let left = rug_uint(bits, 42);
        let right = rug_uint(bits, 1_337);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let product = Integer::from(Mul::mul(black_box(&left), black_box(&right)));
            let _output = black_box(Integer::from(Rem::rem(&product, black_box(&modulus))));
        });
    }
}

/// `a^e mod m` with a full-width exponent.
///
/// The heaviest cell in the suite: one modular multiplication per exponent bit,
/// and the exponent is as wide as the modulus. It runs on the heavy sampling
/// tier for that reason.
mod pow_mod {
    use super::*;

    #[divan::bench(args = MODULAR_EXP, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let base = mp_uint(bits, 42);
        let exponent = mp_uint(bits, 1_337);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&base).pow_mod(black_box(&exponent), black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR_EXP, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let base = rug_uint(bits, 42);
        let exponent = rug_uint(bits, 1_337);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let incomplete = black_box(&base)
                .pow_mod_ref(black_box(&exponent), black_box(&modulus))
                .expect("the positive modulus is nonzero");
            let _output = black_box(Integer::from(incomplete));
        });
    }
}

/// `a^-1 mod m` on an operand guaranteed invertible and not a unit.
mod invert {
    use super::*;

    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let (value_hex, modulus_hex) = coprime_hex_pair(bits);
        let value = MpUint::from_str_radix(&value_hex, 16)
            .expect("generated hexadecimal must parse as MpUint");
        let modulus = MpUint::from_str_radix(&modulus_hex, 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).invert(black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let (value_hex, modulus_hex) = coprime_hex_pair(bits);
        let value = Integer::from_str_radix(&value_hex, 16)
            .expect("generated hexadecimal must parse as Rug Integer");
        let modulus = Integer::from_str_radix(&modulus_hex, 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let incomplete = black_box(&value).invert_ref(black_box(&modulus));
            let _output = black_box(incomplete.map(Integer::from));
        });
    }
}

/// One Montgomery multiplication including domain setup.
///
/// No `rug` half: GMP does not expose its Montgomery representation, so there
/// is nothing to call. The comparable GMP figure is [`mul_mod`], which reaches
/// the same result by reduction instead.
mod montgomery_mul {
    use super::*;

    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let left = mp_uint(bits, 42);
        let right = mp_uint(bits, 1_337);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&left).montgomery_mul(black_box(&right), black_box(&modulus)));
        });
    }
}

/// Barrett reduction of a value against a modulus.
///
/// The counterpart is the plain remainder, which is the operation Barrett
/// replaces.
mod barrett_reduce {
    use super::*;

    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        let modulus = MpUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpUint");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).barrett_reduce(black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Rem::rem(
                black_box(&value),
                black_box(&modulus),
            )));
        });
    }
}
