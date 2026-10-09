//! Factorial, Jacobi, and totient with applicable Rug/GMP and FLINT references.

#![cfg_attr(
    all(
        feature = "_internal-tune",
        target_arch = "x86_64",
        target_os = "linux",
        target_pointer_width = "64"
    ),
    expect(
        unsafe_code,
        reason = "positive odd Jacobi fixtures establish the FLINT reference domain before timing"
    )
)]

use divan::black_box;
use mp_anafis::{MpUint, Precision};
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
use crate::int::support::{FlintInt, flint_uint, pin_flint_to_one_thread, verify_flint_matches_mp};
use crate::int::{
    ladders::EXTENDED_GCD,
    support::{
        SAMPLE_COUNT_GCD, SAMPLE_COUNT_HEAVY, SAMPLE_SIZE_GCD, SAMPLE_SIZE_HEAVY, SAMPLES, measure,
        mp_uint, odd_hex, paired_bench,
    },
};

#[derive(Clone, Copy)]
enum TotientShape {
    Prime,
    Square,
    BalancedSemiprime,
    SmallFactorSemiprime,
}

paired_bench!(factorial, [100, 500, 1_000, 5_000], samples = (4, 30),
    mp: |terms| vec![u32::try_from(terms).expect("factorial argument fits u32")]
        => |n: &u32| MpUint::factorial(*n, Precision::Unlimited),
    rug: |terms| vec![u32::try_from(terms).expect("factorial argument fits u32")]
        => |n: &u32| Integer::from(Integer::factorial(*n)),
);
paired_bench!(jacobi_symbol, EXTENDED_GCD, samples = (SAMPLE_SIZE_GCD, SAMPLE_COUNT_GCD),
    mp: |bits| {
        (0..SAMPLES)
            .map(|index| {
                (
                    mp_uint(bits, 42_u32.wrapping_add(index)),
                    MpUint::from_str_radix(&odd_hex(bits, 9_999_u32.wrapping_add(index)), 16)
                        .expect("odd modulus parses"),
                )
            })
            .collect::<Vec<_>>()
    } => |(a, b): &(MpUint, MpUint)| a.jacobi_symbol(b).map(i32::from),
    rug: |bits| {
        (0..SAMPLES)
            .map(|index| {
                (
                    rug_uint(bits, 42_u32.wrapping_add(index)),
                    Integer::from_str_radix(&odd_hex(bits, 9_999_u32.wrapping_add(index)), 16)
                        .expect("odd modulus parses"),
                )
            })
            .collect::<Vec<_>>()
    } => |(a, b): &(Integer, Integer)| Some(a.jacobi(b)),
    flint: |bits| {
        pin_flint_to_one_thread();
        (0..SAMPLES)
            .map(|index| (
                flint_uint(bits, 42_u32.wrapping_add(index)),
                FlintInt::from_str_radix(&odd_hex(bits, 9_999_u32.wrapping_add(index)), 16),
            ))
            .collect::<Vec<_>>()
    } => |(a, b): &(FlintInt, FlintInt)| {
        // SAFETY: odd_hex and the GCD scenario fixtures construct positive
        // odd moduli; both FLINT owners remain initialized under shared borrows.
        Some(unsafe { a.jacobi_symbol_odd(b) })
    },
    scenarios = crate::int::support::gcd_scenarios,
);

/// Declares one factor distribution with identical prepared batches and timing.
macro_rules! totient_case {
    ($name:ident, $widths:expr, $shape:ident) => {
        mod $name {
            #[cfg(all(
                feature = "_internal-tune",
                target_arch = "x86_64",
                target_os = "linux",
                target_pointer_width = "64"
            ))]
            use super::{FlintInt, verify_totient_inputs};
            use super::{MpUint, SAMPLE_COUNT_HEAVY, TotientShape, measure, totient_inputs};

            #[divan::bench(args = $widths, sample_size = 1, sample_count = SAMPLE_COUNT_HEAVY)]
            fn mp(bencher: divan::Bencher, bits: usize) {
                let inputs = totient_inputs(bits, TotientShape::$shape);
                #[cfg(all(
                    feature = "_internal-tune",
                    target_arch = "x86_64",
                    target_os = "linux",
                    target_pointer_width = "64"
                ))]
                verify_totient_inputs(&inputs);
                measure(bencher, &inputs, MpUint::euler_phi);
            }

            #[cfg(all(
                feature = "_internal-tune",
                target_arch = "x86_64",
                target_os = "linux",
                target_pointer_width = "64"
            ))]
            #[divan::bench(args = $widths, sample_size = 1, sample_count = SAMPLE_COUNT_HEAVY)]
            fn flint(bencher: divan::Bencher, bits: usize) {
                let operands = totient_inputs(bits, TotientShape::$shape);
                verify_totient_inputs(&operands);
                let inputs: Vec<_> = operands
                    .iter()
                    .map(|value| FlintInt::from_str_radix(&value.to_string_radix(16), 16))
                    .collect();
                measure(bencher, &inputs, FlintInt::euler_phi);
            }
        }
    };
}

/// GMP has no totient operation. Random inputs are bounded at 64 bits;
/// explicit factor distributions have separate registered ladders.
mod euler_phi {
    #[expect(
        clippy::wildcard_imports,
        reason = "The paired benchmark uses its category imports"
    )]
    use super::*;

    #[divan::bench(args = [32, 64], sample_size = SAMPLE_SIZE_HEAVY, sample_count = SAMPLE_COUNT_HEAVY)]
    fn mp(bencher: divan::Bencher, bits: usize) {
        let value = mp_uint(bits, 42);
        #[cfg(all(
            feature = "_internal-tune",
            target_arch = "x86_64",
            target_os = "linux",
            target_pointer_width = "64"
        ))]
        verify_totient_inputs(core::slice::from_ref(&value));
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
        let value = flint_uint(bits, 42);
        let mp = mp_uint(bits, 42);
        verify_totient_inputs(core::slice::from_ref(&mp));
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
    fn verify_totient_inputs(inputs: &[MpUint]) {
        pin_flint_to_one_thread();
        for value in inputs {
            let peer = FlintInt::from_str_radix(&value.to_string_radix(16), 16);
            verify_flint_matches_mp(value, &peer);
            verify_flint_matches_mp(
                &value
                    .euler_phi()
                    .expect("constructed factors fit the budget"),
                &peer.euler_phi(),
            );
        }
    }

    totient_case!(prime, [32, 48, 64, 128, 256], Prime);
    totient_case!(square, [32, 64, 128, 256, 512, 1_024], Square);
    totient_case!(balanced_semiprime, [32, 48, 64], BalancedSemiprime);
    totient_case!(
        small_factor_semiprime,
        [64, 96, 128, 256, 512, 1_024],
        SmallFactorSemiprime
    );
}

/// Produces known factor distributions without timing prime searches or setup.
fn totient_inputs(bits: usize, shape: TotientShape) -> Vec<MpUint> {
    (0..SAMPLES)
        .map(|index| {
            let seed = 42_u32.wrapping_add(index.wrapping_mul(1_979));
            let factor_bits = match shape {
                TotientShape::Prime => bits,
                TotientShape::Square | TotientShape::BalancedSemiprime => bits >> 1,
                TotientShape::SmallFactorSemiprime => {
                    bits.checked_sub(16).expect("input wider than small factor")
                }
            };
            // Prefix 110 leaves room for prime search while ensuring two
            // factor prefixes multiply to the requested total bit width.
            let prefix =
                MpUint::from(3_u8) << factor_bits.checked_sub(2).expect("factor width >= 16");
            let candidate = (mp_uint(factor_bits, seed) >> 3_usize) | &prefix;
            let first = candidate.next_prime().expect("positive prime search");
            let value = match shape {
                TotientShape::Prime => first,
                TotientShape::Square => &first * &first,
                TotientShape::BalancedSemiprime => {
                    let other = ((mp_uint(factor_bits, seed.wrapping_add(1_337)) >> 3_usize)
                        | prefix)
                        .next_prime()
                        .expect("positive prime search");
                    let second = if other == first {
                        other.next_prime().expect("distinct prime search")
                    } else {
                        other
                    };
                    first * second
                }
                TotientShape::SmallFactorSemiprime => {
                    let small = MpUint::from(50_021_u32.wrapping_add(index.wrapping_mul(100)))
                        .next_prime()
                        .expect("small prime search");
                    first * small
                }
            };
            assert_eq!(
                value.significant_bits(),
                bits,
                "totient input has its registered width"
            );
            value
        })
        .collect()
}
