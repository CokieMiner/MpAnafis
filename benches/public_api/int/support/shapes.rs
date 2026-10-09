//! Coprime, prime, semiprime, and square-derived deterministic fixtures.
//! Both engines use the same seed streams and mathematical constructions.

use mp_anafis::MpUint;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use rug::Integer;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use super::rug_uint;
use super::{SAMPLES, mp_uint, odd_hex, random_hex};

/// Returns [`SAMPLES`] hexadecimal pairs with `1 < a < m-1` and `gcd(a,m)=1`.
/// The nontrivial units exclude the two self-inverse boundary values. Both
/// engines parse the same generated strings before timing.
#[must_use]
pub fn coprime_hex_pairs(bits: usize) -> Vec<(String, String)> {
    let one = MpUint::one();
    (0..SAMPLES)
        .map(|index| {
            let modulus_hex = odd_hex(bits, 9_999_u32.wrapping_add(index.wrapping_mul(101)));
            let modulus = MpUint::from_str_radix(&modulus_hex, 16)
                .expect("generated odd hexadecimal must parse as MpUint");
            let mut candidate_seed = 42_u32.wrapping_add(index.wrapping_mul(1_979));

            loop {
                let candidate_hex = random_hex(bits, candidate_seed);
                let candidate = MpUint::from_str_radix(&candidate_hex, 16)
                    .expect("generated hexadecimal must parse as MpUint");
                let successor = candidate
                    .checked_add(&one)
                    .expect("unlimited precision addition never overflows");
                let is_representative =
                    candidate > one && candidate < modulus && successor != modulus;
                if is_representative && candidate.gcd(&modulus).is_one() {
                    return (candidate_hex, modulus_hex);
                }
                candidate_seed = candidate_seed.wrapping_add(1);
            }
        })
        .collect()
}

/// Primes obtained by exclusive next-prime search from the fixed seed stream.
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

/// Products of two prime factors whose seed widths are at least 32 bits.
#[expect(
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
#[expect(
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

/// Exact squares of deterministic half-width roots, with a 32-bit minimum root.
#[expect(
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
#[expect(
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

/// Nonsquares of the form `r^2+1`, with positive deterministic roots.
/// Residue screens may reject these values before root computation.
#[expect(
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
#[expect(
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
