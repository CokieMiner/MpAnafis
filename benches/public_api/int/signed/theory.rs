//! Signed number theory and modular arithmetic.
//!
//! Every one of these delegates to the `MpUint` implementation after
//! normalising the sign into the operation's domain, so the gap against the
//! matching [`unsigned`](crate::int::unsigned) cell is the normalisation.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
use mp_anafis::MpInt;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::{rug_int, rug_int_pairs};
use crate::int::{
    ladders::{EXTENDED_GCD, MODULAR_EXP, NARROW, THEORY},
    support::{
        SAMPLE_COUNT_HEAVY, SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST, SAMPLE_SIZE_HEAVY,
        SAMPLE_SIZE_WIDE, mp_int, mp_int_pairs, odd_hex,
    },
};

/// GCD of two negative values, which is defined on the magnitudes.
mod gcd {
    use super::*;

    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).gcd(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, true);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(black_box(left).gcd_ref(black_box(right))));
            }
        });
    }
}

mod extended_gcd {
    use super::*;

    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_int_pairs(bits, true, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).extended_gcd(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_int_pairs(bits, true, false);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let cofactor = Integer::new();
                let _output = black_box(
                    black_box(left)
                        .clone()
                        .extended_gcd(black_box(right).clone(), cofactor),
                );
            }
        });
    }
}

/// `a^e mod m` with a negative base, which must be reduced into the residue
/// class before exponentiation.
mod pow_mod {
    use super::*;

    #[divan::bench(args = MODULAR_EXP, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let base = mp_int(bits, 42, true);
        let exponent = mp_int(bits, 1_337, false);
        let modulus = MpInt::from_str_radix(&odd_hex(bits, 9_999), 16)
            .expect("generated modulus must parse as MpInt");
        bencher.bench_local(|| {
            let _output =
                black_box(black_box(&base).pow_mod(black_box(&exponent), black_box(&modulus)));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = MODULAR_EXP, sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let base = rug_int(bits, 42, true);
        let exponent = rug_int(bits, 1_337, false);
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

/// The square root, which rejects negative input before doing any work.
mod checked_isqrt {
    use super::*;

    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_int(bits, 42, false);
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).checked_isqrt());
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = NARROW, sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = rug_int(bits, 42, false);
        bencher.bench_local(|| {
            let _output = black_box(Integer::from(black_box(&value).sqrt_ref()));
        });
    }
}

/// Primality of a positive `MpInt`, which is the unsigned test behind a sign
/// check.
mod is_probably_prime {
    use super::*;

    const ROUNDS: u32 = 24;

    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = MpInt::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as MpInt");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_probably_prime(ROUNDS));
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = [256, 1_024], sample_size = SAMPLE_SIZE_FAST)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let value = Integer::from_str_radix(&odd_hex(bits, 42), 16)
            .expect("generated odd hexadecimal must parse as Rug Integer");
        bencher.bench_local(|| {
            let _output = black_box(black_box(&value).is_probably_prime(ROUNDS));
        });
    }
}
