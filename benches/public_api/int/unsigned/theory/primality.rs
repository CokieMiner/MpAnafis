//! Primality testing and prime search.
//!
//! Three input shapes, because they reach different amounts of the algorithm: a
//! random odd number usually falls to the first witness, a prime runs every
//! requested round, and a semiprime with no small factor defeats the
//! trial-division screen before the witness loop even starts.
//!
//! Everything here runs on the heavy sampling tier. One call is already
//! hundreds of modular exponentiations, and `next_prime` is hundreds of those
//! calls: at the fast tier a single 1024-bit cell does not finish in an hour.
//!
//! The opt-in FLINT cells are explicitly named cost references. FLINT's
//! `fmpz_is_probabprime` uses its own fixed policy and cannot be asked to run
//! the same 24 Miller-Rabin rounds as Mp and GMP/Rug.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use crate::int::support::{FlintInt, pin_flint_to_one_thread, verify_flint_matches_mp};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_known_primes, rug_semiprimes_no_small_factors};
use crate::int::{
    ladders::PRIMALITY,
    support::{
        SAMPLE_COUNT_HEAVY, SAMPLE_SIZE_HEAVY, mp_known_primes, mp_semiprimes_no_small_factors,
        odd_hex,
    },
};

/// Miller-Rabin rounds requested. GMP and we both interpret this as a witness
/// count, so the two sides do the same amount of work.
const ROUNDS: u32 = 24;

mod is_probably_prime_random {
    use super::*;

    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = MpUint::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as MpUint");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_probably_prime(ROUNDS));
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn flint_fixed_policy_cost_reference(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let text = odd_hex(bits, 42);
        let mp_value = MpUint::from_str_radix(&text, 16)
            .expect("generated odd hexadecimal must parse as MpUint");
        let value = FlintInt::from_str_radix(&text, 16);
        verify_flint_matches_mp(&mp_value, &value);
        assert_eq!(
            value.probable_prime_cost_reference(),
            mp_value.is_probably_prime(ROUNDS),
            "FLINT and Mp must classify the random odd input identically"
        );
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).probable_prime_cost_reference());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = Integer::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as Rug Integer");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_probably_prime(ROUNDS));
        });
    }
}

mod is_probably_prime_known_prime {
    use super::*;

    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let values = mp_known_primes(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_probably_prime(ROUNDS));
            }
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn flint_fixed_policy_cost_reference(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let values = mp_known_primes(bits);
        let flint_values: Vec<_> = values
            .iter()
            .map(|value| FlintInt::from_str_radix(&format!("{value:x}"), 16))
            .collect();
        for (value, flint_value) in values.iter().zip(&flint_values) {
            verify_flint_matches_mp(value, flint_value);
            assert_eq!(
                flint_value.probable_prime_cost_reference(),
                value.is_probably_prime(ROUNDS),
                "FLINT and Mp must classify known primes identically"
            );
        }
        bencher.bench_local(|| {
            for value in &flint_values {
                let _output = black_box(black_box(value).probable_prime_cost_reference());
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let values = rug_known_primes(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_probably_prime(ROUNDS));
            }
        });
    }
}

mod is_probably_prime_semiprime {
    use super::*;

    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let values = mp_semiprimes_no_small_factors(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_probably_prime(ROUNDS));
            }
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn flint_fixed_policy_cost_reference(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let values = mp_semiprimes_no_small_factors(bits);
        let flint_values: Vec<_> = values
            .iter()
            .map(|value| FlintInt::from_str_radix(&format!("{value:x}"), 16))
            .collect();
        for (value, flint_value) in values.iter().zip(&flint_values) {
            verify_flint_matches_mp(value, flint_value);
            assert_eq!(
                flint_value.probable_prime_cost_reference(),
                value.is_probably_prime(ROUNDS),
                "FLINT and Mp must classify semiprimes identically"
            );
        }
        bencher.bench_local(|| {
            for value in &flint_values {
                let _output = black_box(black_box(value).probable_prime_cost_reference());
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let values = rug_semiprimes_no_small_factors(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_probably_prime(ROUNDS));
            }
        });
    }
}

/// The deterministic test, which has no bounded-round GMP equivalent.
///
/// The counterpart is `is_probably_prime` at GMP's own deterministic-enough
/// round count; the two answer the same question with different certainty, so
/// this cell is a cost comparison rather than an equivalence.
mod is_prime {
    use super::*;

    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let values = mp_known_primes(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_prime());
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let values = rug_known_primes(bits);
        bencher.bench_local(|| {
            for value in &values {
                let _output = black_box(black_box(value).is_probably_prime(ROUNDS));
            }
        });
    }
}

mod next_prime {
    use super::*;

    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = MpUint::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as MpUint");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).next_prime());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = PRIMALITY, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = Integer::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as Rug Integer");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).clone().next_prime());
        });
    }
}
