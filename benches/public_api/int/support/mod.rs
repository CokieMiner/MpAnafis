//! Deterministic operand generation and sampling policy shared by the integer
//! benchmarks.
//!
//! Operands are derived from a fixed linear congruential stream keyed by a seed,
//! so an `mp` function and its `rug` counterpart that pass the same seed and
//! bit width receive numerically identical inputs. Nothing here is timed; every
//! generator runs outside the `bencher.bench_local` closure.

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
mod flint;
mod operands;
mod shapes;
mod verification;

#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use flint::{FlintInt, pin_flint_to_one_thread};
pub use operands::{
    bounded_mp_uint, mp_int, mp_int_pairs, mp_uint, mp_uint_pairs, mp_uint_pairs_with_widths,
    odd_hex, random_hex,
};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use operands::{flint_odd_uint, flint_uint, flint_uint_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use operands::{rug_int, rug_int_pairs, rug_uint, rug_uint_pairs, rug_uint_pairs_with_widths};
pub use shapes::{
    coprime_hex_pair, mp_div_pairs_2n_n, mp_div_pairs_3n2_n, mp_div_pairs_same_limbs_ge_2,
    mp_known_primes, mp_semiprimes_no_small_factors, mp_square_plus_one, mp_true_squares,
};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use shapes::{
    rug_div_pairs_2n_n, rug_div_pairs_3n2_n, rug_div_pairs_same_limbs_ge_2, rug_known_primes,
    rug_semiprimes_no_small_factors, rug_square_plus_one, rug_true_squares,
};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use verification::verify_flint_matches_mp;
pub use verification::{verify_mp_int_division_pairs, verify_mp_uint_division_pairs};

/// Operands per generated batch.
///
/// Benchmarks that loop over a batch report the time for all [`SAMPLES`]
/// operations, not one; ratios against the paired `rug` function are unaffected
/// because both sides loop over batches of the same size.
pub const SAMPLES: u32 = 10;

// ============================================================================
// Sampling
// ============================================================================
//
// Every benchmark in this target pins `sample_size`, because divan's automatic
// tuning is not safe for operations this fast.
//
// Tuning doubles the sample size until the *slowest* trial sample exceeds a
// hundred times the timer precision. A single interrupted trial therefore ends
// tuning immediately, and the benchmark collects at `sample_size = 1`: one
// operation of a few nanoseconds, timed against a ten-nanosecond timer. Every
// sample is then noise, so the median is noise too -- this is not an outlier
// that more samples would average away.
//
// It is sporadic and can strike any cell, and it has produced published-looking
// numbers that were wrong by a factor of three: a 4-limb multiplication that
// appeared 3x slower than GMP measured dead level once pinned, and a 128-bit
// square that appeared 2.7x behind was 0.88x. Pinning the size removes the
// tuning phase entirely.
//
// Three tiers, because one value cannot serve the whole range. The distinction
// is the *slowest* argument a benchmark runs, not the fastest.

/// For benchmarks whose widest argument stays at or below 4096 bits.
///
/// At roughly 40 ns per iteration this puts a sample near ten microseconds,
/// comfortably clear of timer precision.
pub const SAMPLE_SIZE_FAST: u32 = 256;

/// For benchmarks that reach 16 Kibit or beyond, where a single iteration can
/// already cost milliseconds.
///
/// Paired with a reduced sample count so total runtime stays bounded: the
/// widest multiplication cell costs about four milliseconds per iteration, and
/// `SAMPLE_SIZE_FAST` there would mean over a minute for one argument. Thirty
/// two iterations still clear the timer by a wide margin at the narrow end of
/// these ladders, which is where the pathology bites.
pub const SAMPLE_SIZE_WIDE: u32 = 32;

/// Sample count for the wide tier, halved to offset the larger sample size.
pub const SAMPLE_COUNT_WIDE: u32 = 50;

/// For single operations costing a millisecond or more on their own, such as
/// wide modular exponentiation.
///
/// `SAMPLE_SIZE_FAST` at a ten millisecond iteration means four minutes for one
/// cell, which is what the 4096-bit `pow_mod` ladder used to cost. Four
/// iterations still clear a twenty nanosecond timer by five orders of
/// magnitude.
pub const SAMPLE_SIZE_HEAVY: u32 = 4;

/// Sample count for the heavy tier.
pub const SAMPLE_COUNT_HEAVY: u32 = 20;
