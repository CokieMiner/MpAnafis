//! `factorial`, `jacobi_symbol`, `euler_phi`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use arbi_anafis::{ArbiUint, Precision};
use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint;
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use crate::int::support::{
    FlintInt, flint_odd_uint, flint_uint, pin_flint_to_one_thread, verify_flint_matches_arbi,
};
use crate::int::support::{
    SAMPLE_COUNT_HEAVY, SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_HEAVY, SAMPLE_SIZE_WIDE,
    arbi_uint, odd_hex,
};

/// `n!`, argued by `n` rather than by a bit width.
mod factorial {
    use super::*;

    const TERMS: [u32; 4] = [100, 500, 1_000, 5_000];

    #[divan::bench(args = TERMS, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn arbi(bencher: divan::Bencher, terms: u32) {
        bencher.bench_local(|| {
            let _output = black_box(ArbiUint::factorial(black_box(terms), Precision::Unlimited));
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = TERMS, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn flint(bencher: divan::Bencher, terms: u32) {
        pin_flint_to_one_thread();
        let expected = ArbiUint::factorial(terms, Precision::Unlimited);
        let actual = FlintInt::factorial(terms);
        verify_flint_matches_arbi(&expected, &actual);
        bencher.bench_local(|| {
            let _output = black_box(FlintInt::factorial(black_box(terms)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = TERMS, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, terms: u32) {
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(Integer::factorial(black_box(terms))));
        });
    }
}

/// The Jacobi symbol against an odd modulus, which is its domain.
mod jacobi_symbol {
    use super::*;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        let modulus = ArbiUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as ArbiUint");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).jacobi_symbol(black_box(&modulus)));
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn flint(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let arbi_value = arbi_uint(bits, 42);
        let arbi_modulus = ArbiUint::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as ArbiUint");
        let value = flint_uint(bits, 42);
        let modulus = flint_odd_uint(bits, 9_999);
        verify_flint_matches_arbi(&arbi_value, &value);
        verify_flint_matches_arbi(&arbi_modulus, &modulus);
        assert_eq!(
            value.jacobi(&modulus),
            i32::from(
                arbi_value
                    .jacobi_symbol(&arbi_modulus)
                    .expect("the generated modulus is positive and odd")
            ),
            "FLINT and Arbi must compute the same Jacobi symbol"
        );
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).jacobi(black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_uint(bits, 42);
        let modulus = Integer::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as Rug Integer");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).jacobi(black_box(&modulus)));
        });
    }
}

/// Euler's totient.
///
/// No `rug` half: GMP has no totient, because computing it requires the
/// factorisation. Instead, we bench against FLINT.
///
/// Which is also why the ladder stops at 64 bits and runs on the heavy tier.
/// This is a factoring benchmark wearing a different name, and factoring cost
/// grows with the size of the *second largest* prime factor, not with the width
/// — a single unlucky 128-bit operand does not finish in an hour, so a cell for
/// it would never report.
mod euler_phi {
    use super::*;

    #[divan::bench(args = [32, 64], sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn arbi(bencher: divan::Bencher, bits: usize) {
        let value = arbi_uint(bits, 42);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).euler_phi());
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = [32, 64], sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn flint(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let arbi_value = arbi_uint(bits, 42);
        let value = flint_uint(bits, 42);
        verify_flint_matches_arbi(&arbi_value, &value);
        let expected = arbi_value
            .euler_phi()
            .expect("positive benchmark input has a totient");
        let actual = value.euler_phi();
        verify_flint_matches_arbi(&expected, &actual);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).euler_phi());
        });
    }
}
