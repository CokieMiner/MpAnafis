//! Shared fixtures, paired declarations, timed loops, and untimed verification.

mod comparison;
mod division;
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
mod flint;
mod gcd;
mod measurement;
mod operands;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod outcome;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod rug_ops;
mod sampling;
mod shapes;
mod templates;
mod verification;

pub use crate::{gcd_scenarios, paired_assign, paired_bench, paired_mutate};

pub use comparison::{clamped, hash_value, maximum, minimum};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use comparison::{verify_hashes, verify_selection};
pub use division::{
    DivisibilityShape, DivisionResidue, DivisionShape, mp_divisibility_pairs, mp_division_pairs,
};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use division::{rug_divisibility_pairs, rug_division_pairs};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use flint::{FlintInt, pin_flint_to_one_thread};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use gcd::flint_gcd_pairs;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use gcd::rug_gcd_pairs;
pub use gcd::{GcdShape, mp_gcd_pairs};
pub use measurement::{measure, measure_assignment, measure_mutation};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use operands::flint_uint;
pub use operands::{
    bounded_mp_int, bounded_mp_uint, mp_int, mp_int_pairs, mp_uint, mp_uint_lopsided_pairs,
    mp_uint_pairs, odd_hex, random_hex,
};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use operands::{rug_int, rug_int_pairs, rug_uint, rug_uint_lopsided_pairs, rug_uint_pairs};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use outcome::Outcome;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use rug_ops::{
    from_signed_be_bytes, nearest_f32, nearest_f64, signed_be_bytes, signed_words, unsigned_words,
};
pub use sampling::{
    SAMPLE_COUNT_FAST, SAMPLE_COUNT_GCD, SAMPLE_COUNT_HEAVY, SAMPLE_COUNT_WIDE, SAMPLE_SIZE_FAST,
    SAMPLE_SIZE_GCD, SAMPLE_SIZE_HEAVY, SAMPLE_SIZE_WIDE, SAMPLES,
};
pub use shapes::{
    coprime_hex_pairs, mp_known_primes, mp_semiprimes_no_small_factors, mp_square_plus_one,
    mp_true_squares,
};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use shapes::{
    rug_known_primes, rug_semiprimes_no_small_factors, rug_square_plus_one, rug_true_squares,
};
#[cfg(all(
    feature = "_internal-tune",
    target_arch = "x86_64",
    target_os = "linux",
    target_pointer_width = "64"
))]
pub use verification::verify_flint_matches_mp;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
pub use verification::{verify_assignment, verify_extended_gcd_pairs, verify_pair};
pub use verification::{verify_mp_int_division_pairs, verify_mp_uint_division_pairs};
