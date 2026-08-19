//! Primality and next-prime operations owned by [`InternalMpUint`].

#![allow(
    unsafe_code,
    reason = "The Miller-Rabin base range ends at min(rounds, 64); sieve remainders divide only by SIEVE_PRIMES entries in 3..=251, so every unchecked divisor is nonzero."
)]

use core::cmp::Ordering;

use super::{InternalMpUint, Limb, MontgomeryDomain, MulScratch};

const SIEVE_WORDS: usize = 32;

/// Namespace for the cross-file primality algorithm surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primality;

pub const SIEVE_PRIMES: [Limb; 53] = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
];
impl InternalMpUint {
    /// Deterministic primality test for inputs <= 2^64.
    ///
    /// For numbers <= 2^64 this uses a deterministic Miller-Rabin test with
    /// Sinclair bases. For larger inputs, this is a **probabilistic** test
    /// using 24 prime bases.
    ///
    /// Returns `false` for 0 and 1.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        if self.is_zero() || self.is_one() {
            return false;
        }
        if self.cmp(&Self::from_limb(3)) == Ordering::Less {
            return true;
        }
        if self.is_even() {
            return false;
        }
        if self.cmp(&Self::from_limb(9)) == Ordering::Less {
            return true;
        }

        // `is_probably_prime` screens by trial division itself and dispatches
        // to the deterministic u64 test, so there is nothing to do here first.
        self.is_probably_prime(24)
    }

    /// Returns `true` when `self` is *probably* prime using `k` rounds of the
    /// Miller-Rabin test with a fixed set of bases.
    ///
    /// This is a probabilistic test - the probability of a composite
    /// passing is < (1/4)^k for large numbers.
    ///
    /// # Panics
    ///
    /// May panic when internal invariants are violated (should not happen for
    /// well-formed inputs).
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        if self.is_zero() || self.is_one() {
            return false;
        }
        if self.is_even() {
            return self.cmp(&Self::from_limb(2)) == Ordering::Equal;
        }
        if let Some(native) = self.to_u64() {
            return Primality::is_prime_u64(native);
        }

        // Reject small-prime multiples before paying for a single modexp. The
        // `to_u64` arm above already returned for everything that fits in 64
        // bits, so `self` here exceeds every prime the screen divides by and a
        // hit is always a proper factor, never `self` itself.
        if !Primality::trial_division(self) {
            return false;
        }

        // Write n-1 = d * 2^s with d odd.
        // SAFETY: n >= 2, so n-1 is non-negative
        let n_minus_1 = self.sub(&Self::one());
        let s = n_minus_1.trailing_zeros();
        let d = n_minus_1.shr(s);

        // SAFETY: self is checked to be odd right before this
        let domain = MontgomeryDomain::new(self);

        let rounds = if k == 0 {
            1
        } else {
            usize::try_from(k).unwrap_or(usize::MAX)
        };

        // 64 prime bases, supporting up to 64 rounds of deterministic Miller-Rabin.
        let prime_bases: [u64; 64] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
            181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
            277, 281, 283, 293, 307, 311,
        ];

        let limit = rounds.min(prime_bases.len());
        // SAFETY: `limit = rounds.min(prime_bases.len())`, so `..limit` is
        // within the initialized 64-element prime-base array.
        let bases_to_check = unsafe { prime_bases.get_unchecked(..limit) };

        let cap = self.limbs().len().wrapping_add(1);
        let mut temp_prod = Self::with_capacity(cap);
        let mut temp_rem = Self::with_capacity(cap);
        let mut mul_scratch = MulScratch::default();
        let n_minus_1_mont =
            domain.transform_into_with_scratch(&n_minus_1, &mut temp_prod, &mut mul_scratch);
        let one_mont =
            domain.transform_into_with_scratch(&Self::one(), &mut temp_prod, &mut mul_scratch);
        for &base in bases_to_check {
            let b = Self::from_u64(base);
            if self.cmp(&b) != Ordering::Greater {
                continue;
            }
            if !Primality::miller_rabin_test(
                &b,
                &d,
                s,
                &n_minus_1_mont,
                &one_mont,
                &mut temp_prod,
                &mut temp_rem,
                &domain,
                &mut mul_scratch,
            ) {
                return false;
            }
        }
        true
    }

    /// Returns the smallest prime >= `self`.
    #[must_use]
    pub fn next_prime(&self) -> Self {
        let mut candidate = self.clone();

        if candidate.cmp(&Self::from_limb(2)) == Ordering::Less {
            return Self::from_limb(2);
        }
        if candidate.is_even() {
            candidate.add_assign(&Self::from_limb(1));
        }

        loop {
            let mut sieve = [0_u64; SIEVE_WORDS];
            let limbs = candidate.limbs();

            for &p in &SIEVE_PRIMES {
                #[allow(clippy::as_conversions, reason = "Limb fits safely in u128")]
                let p_128 = p as u128;
                let mut r = 0_u128;
                #[allow(clippy::as_conversions, reason = "Limb fits safely in u128")]
                for &limb in limbs.iter().rev() {
                    let shifted = r.wrapping_shl(Limb::BITS) | (limb as u128);
                    // SAFETY: p is from SIEVE_PRIMES (starts at 3), guaranteed non-zero.
                    r = unsafe { shifted.checked_rem(p_128).unwrap_unchecked() };
                }
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "remainder fits in Limb"
                )]
                let rem = r as Limb;

                let target = if rem == 0 { 0 } else { p.wrapping_sub(rem) };
                let inv2 = p.wrapping_add(1) >> 1;
                // SAFETY: `p` comes from `SIEVE_PRIMES`, whose smallest entry
                // is 3, so the checked remainder cannot return `None`.
                let mut start_i =
                    unsafe { target.wrapping_mul(inv2).checked_rem(p).unwrap_unchecked() };

                let single_candidate = if limbs.len() == 1 {
                    limbs.first().copied()
                } else {
                    None
                };
                if let Some(candidate_value) = single_candidate
                    && candidate_value <= p
                {
                    let prime_offset = p.wrapping_sub(candidate_value);
                    let candidate_index = prime_offset >> 1;
                    if prime_offset & 1 == 0 && start_i == candidate_index {
                        start_i = start_i.wrapping_add(p);
                    }
                }

                let mut i = start_i;
                let limit = SIEVE_WORDS.wrapping_mul(64);
                while i < limit {
                    if let Some(word) = sieve.get_mut(i >> 6) {
                        *word |= 1_u64 << (i & 63);
                    }
                    i = i.wrapping_add(p);
                }
            }

            let limit = SIEVE_WORDS.wrapping_mul(64);
            for i in 0..limit {
                if let Some(&word) = sieve.get(i >> 6)
                    && (word & (1_u64 << (i & 63))) == 0
                {
                    let mut test_cand = candidate.clone();
                    if i > 0 {
                        let i_limb = i;
                        test_cand.add_assign(&Self::from_limb(i_limb.wrapping_shl(1)));
                    }
                    if test_cand.is_prime() {
                        return test_cand;
                    }
                }
            }

            let step_limb = SIEVE_WORDS * 128;
            candidate.add_assign(&Self::from_limb(step_limb));
        }
    }
}
