//! Operand *shapes*: batches whose structure, not just width, selects the code
//! path under test.
//!
//! A random equal-width division pair has a one-limb quotient and never reaches
//! the recursive divider; a semiprime defeats the trial-division screen a random
//! odd number falls out of; a true square is the only input on which
//! `is_perfect_square` runs to completion. Timing an algorithm on the shape that
//! skips it measures the guard clause, so each shape here exists to force one
//! specific path.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

use super::{SAMPLES, mp_uint, mp_uint_pairs_with_widths, odd_hex, random_hex};
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::{rug_uint, rug_uint_pairs_with_widths};

// ---------------------------------------------------------------------------
// Division shapes
// ---------------------------------------------------------------------------

/// A `2n / n` batch: the balanced recursive-division shape.
#[must_use]
pub fn mp_div_pairs_2n_n(bits: usize) -> Vec<(MpUint, MpUint)> {
    mp_uint_pairs_with_widths(bits.saturating_mul(2), bits)
}

/// The Rug counterpart of [`mp_div_pairs_2n_n`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_div_pairs_2n_n(bits: usize) -> Vec<(Integer, Integer)> {
    rug_uint_pairs_with_widths(bits.saturating_mul(2), bits)
}

/// A `3n/2 / n` batch: the odd-split shape that exercises the recursive
/// divider's uneven branch.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark bit width calculations are exact and guaranteed not to overflow"
)]
#[must_use]
pub fn mp_div_pairs_3n2_n(bits: usize) -> Vec<(MpUint, MpUint)> {
    mp_uint_pairs_with_widths(bits.saturating_add(bits >> 1), bits)
}

/// The Rug counterpart of [`mp_div_pairs_3n2_n`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark bit width calculations are exact and guaranteed not to overflow"
)]
#[must_use]
pub fn rug_div_pairs_3n2_n(bits: usize) -> Vec<(Integer, Integer)> {
    rug_uint_pairs_with_widths(bits.saturating_add(bits >> 1), bits)
}

/// An equal-limb batch whose divisor is shifted down two bits, forcing a
/// quotient of at least two.
///
/// The plain equal-width shape yields a quotient of one, where the whole of
/// division reduces to a compare and a subtract.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn mp_div_pairs_same_limbs_ge_2(bits: usize) -> Vec<(MpUint, MpUint)> {
    (0..SAMPLES)
        .map(|index| {
            let left = mp_uint(bits, 42_u32.wrapping_add(index));
            let mut right = mp_uint(bits, 1_337_u32.wrapping_add(index));
            right >>= 2;
            if right.is_zero() {
                right = MpUint::from(1_u32);
            }
            (left, right)
        })
        .collect()
}

/// The Rug counterpart of [`mp_div_pairs_same_limbs_ge_2`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn rug_div_pairs_same_limbs_ge_2(bits: usize) -> Vec<(Integer, Integer)> {
    (0..SAMPLES)
        .map(|index| {
            let left = rug_uint(bits, 42_u32.wrapping_add(index));
            let mut right = rug_uint(bits, 1_337_u32.wrapping_add(index));
            right >>= 2;
            if right == 0 {
                right = Integer::from(1);
            }
            (left, right)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Modular shapes
// ---------------------------------------------------------------------------

/// Returns `(value_hex, modulus_hex)` where the value is invertible modulo the
/// modulus and is neither of the two units.
///
/// `invert` returns early for `1` and for `modulus - 1`, which are their own
/// inverses, so both are excluded: an operand that hits either would measure the
/// guard clause rather than the extended GCD. The search is over `MpUint` so
/// the pair is available on targets without Rug, and both libraries parse the
/// same two strings.
#[must_use]
pub fn coprime_hex_pair(bits: usize) -> (String, String) {
    let modulus_hex = odd_hex(bits, 9_999);
    let modulus = MpUint::from_str_radix(&modulus_hex, 16)
        .expect("generated odd hexadecimal must parse as MpUint");
    let one = MpUint::one();
    let mut candidate_seed = 42_u32;

    loop {
        let candidate_hex = random_hex(bits, candidate_seed);
        let candidate = MpUint::from_str_radix(&candidate_hex, 16)
            .expect("generated hexadecimal must parse as MpUint");
        let successor = candidate
            .checked_add(&one)
            .expect("unlimited precision addition never overflows");
        let is_representative = candidate > one && candidate < modulus && successor != modulus;
        if is_representative && candidate.gcd(&modulus).is_one() {
            return (candidate_hex, modulus_hex);
        }
        candidate_seed = candidate_seed.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// Primality shapes
// ---------------------------------------------------------------------------

/// A batch of primes: the worst case for Miller-Rabin, where every requested
/// round runs to completion.
#[must_use]
pub fn mp_known_primes(bits: usize) -> Vec<MpUint> {
    (0..SAMPLES)
        .map(|index| {
            mp_uint(bits, 42_u32.wrapping_add(index))
                .next_prime()
                .expect("next_prime returns Some for valid benchmark bit widths")
        })
        .collect()
}

/// The Rug counterpart of [`mp_known_primes`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[must_use]
pub fn rug_known_primes(bits: usize) -> Vec<Integer> {
    (0..SAMPLES)
        .map(|index| rug_uint(bits, 42_u32.wrapping_add(index)).next_prime())
        .collect()
}

/// A batch of semiprimes with two half-width prime factors.
///
/// Composite, but with no small factor for trial division to find, so the
/// witness loop is entered before the answer is known.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn mp_semiprimes_no_small_factors(bits: usize) -> Vec<MpUint> {
    let half_bits = bits >> 1;
    (0..SAMPLES)
        .map(|index| {
            let p1 = mp_uint(half_bits.max(32), 42_u32.wrapping_add(index))
                .next_prime()
                .expect("next_prime returns Some for valid benchmark bit widths");
            let p2 = mp_uint(half_bits.max(32), 1_337_u32.wrapping_add(index))
                .next_prime()
                .expect("next_prime returns Some for valid benchmark bit widths");
            &p1 * &p2
        })
        .collect()
}

/// The Rug counterpart of [`mp_semiprimes_no_small_factors`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn rug_semiprimes_no_small_factors(bits: usize) -> Vec<Integer> {
    let half_bits = bits >> 1;
    (0..SAMPLES)
        .map(|index| {
            let p1 = rug_uint(half_bits.max(32), 42_u32.wrapping_add(index)).next_prime();
            let p2 = rug_uint(half_bits.max(32), 1_337_u32.wrapping_add(index)).next_prime();
            Integer::from(&p1 * &p2)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Perfect square shapes
// ---------------------------------------------------------------------------

/// A batch of exact squares: `is_perfect_square` must compute the full root.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn mp_true_squares(bits: usize) -> Vec<MpUint> {
    let half_bits = bits >> 1;
    (0..SAMPLES)
        .map(|index| {
            let root = mp_uint(half_bits.max(32), 42_u32.wrapping_add(index));
            &root * &root
        })
        .collect()
}

/// The Rug counterpart of [`mp_true_squares`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn rug_true_squares(bits: usize) -> Vec<Integer> {
    let half_bits = bits >> 1;
    (0..SAMPLES)
        .map(|index| {
            let root = rug_uint(half_bits.max(32), 42_u32.wrapping_add(index));
            Integer::from(&root * &root)
        })
        .collect()
}

/// A batch of near misses that pass the residue screen but fail the root.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn mp_square_plus_one(bits: usize) -> Vec<MpUint> {
    mp_true_squares(bits)
        .into_iter()
        .map(|mut square| {
            square += MpUint::from(1_u32);
            square
        })
        .collect()
}

/// The Rug counterpart of [`mp_square_plus_one`].
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "benchmark operand construction is guaranteed not to overflow"
)]
#[must_use]
pub fn rug_square_plus_one(bits: usize) -> Vec<Integer> {
    rug_true_squares(bits)
        .into_iter()
        .map(|mut square| {
            square += 1;
            square
        })
        .collect()
}
