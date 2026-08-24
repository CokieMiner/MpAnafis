//! `gcd`, `lcm`, `gcd_lcm`, `extended_gcd`, `is_coprime`.

#![allow(
    clippy::wildcard_imports,
    reason = "benchmark submodules inherit parent scope"
)]

use divan::black_box;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use crate::int::support::rug_uint_pairs;
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
use crate::int::support::{flint_uint_pairs, pin_flint_to_one_thread, verify_flint_matches_mp};
use crate::int::{
    ladders::{EXTENDED_GCD, THEORY},
    support::{SAMPLE_COUNT_WIDE, SAMPLE_SIZE_WIDE, mp_uint_pairs},
};

mod gcd {
    use super::*;

    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).gcd(black_box(right)));
            }
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn flint(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let inputs = flint_uint_pairs(bits);
        let mp_inputs = mp_uint_pairs(bits);
        assert_eq!(inputs.len(), mp_inputs.len(), "paired benchmark batch");
        for ((left, right), (mp_left, mp_right)) in inputs.iter().zip(&mp_inputs) {
            verify_flint_matches_mp(mp_left, left);
            verify_flint_matches_mp(mp_right, right);
            let actual = left.gcd(right);
            let expected = mp_left.gcd(mp_right);
            verify_flint_matches_mp(&expected, &actual);
        }
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).gcd(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(black_box(left).gcd_ref(black_box(right))));
            }
        });
    }
}

mod lcm {
    use super::*;

    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).lcm(black_box(right)));
            }
        });
    }

    #[cfg(all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn flint(bencher: divan::Bencher, bits: usize) {
        pin_flint_to_one_thread();
        let inputs = flint_uint_pairs(bits);
        let mp_inputs = mp_uint_pairs(bits);
        assert_eq!(inputs.len(), mp_inputs.len(), "paired benchmark batch");
        for ((left, right), (mp_left, mp_right)) in inputs.iter().zip(&mp_inputs) {
            verify_flint_matches_mp(mp_left, left);
            verify_flint_matches_mp(mp_right, right);
            let actual = left.lcm(right);
            let expected = mp_left
                .lcm(mp_right)
                .expect("nonzero benchmark operands have a representable LCM");
            verify_flint_matches_mp(&expected, &actual);
        }
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).lcm(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(Integer::from(black_box(left).lcm_ref(black_box(right))));
            }
        });
    }
}

/// Both results from one traversal.
///
/// The counterpart computes them separately, which is all GMP offers, so the
/// cell shows what sharing the traversal is worth.
mod gcd_lcm {
    use super::*;

    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).gcd_lcm(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let divisor = Integer::from(black_box(left).gcd_ref(black_box(right)));
                let multiple = Integer::from(black_box(left).lcm_ref(black_box(right)));
                let _output = black_box((divisor, multiple));
            }
        });
    }
}

/// The Bezout coefficients, which modular inversion is built on.
mod extended_gcd {
    use super::*;

    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).extended_gcd(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = EXTENDED_GCD, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
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

/// Whether the GCD is one, which need not produce the GCD itself.
mod is_coprime {
    use super::*;

    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let inputs = mp_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let _output = black_box(black_box(left).is_coprime(black_box(right)));
            }
        });
    }

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    #[divan::bench(args = THEORY, sample_size = SAMPLE_SIZE_WIDE, sample_count = SAMPLE_COUNT_WIDE)]
    fn rug(bencher: divan::Bencher, bits: usize) {
        let inputs = rug_uint_pairs(bits);
        bencher.bench_local(|| {
            for (left, right) in &inputs {
                let divisor = Integer::from(black_box(left).gcd_ref(black_box(right)));
                let _output = black_box(divisor == 1_u32);
            }
        });
    }
}
