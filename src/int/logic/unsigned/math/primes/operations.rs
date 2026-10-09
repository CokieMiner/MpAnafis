//! Primality and strict prime searches owned by [`InternalMpUint`].

#![expect(
    unsafe_code,
    reason = "The fixed prime-base prefix is bounded by its table."
)]

use core::cmp::Ordering;

use super::{InternalMpUint, MontgomeryDomain, MontgomeryScratch};

/// Namespace for the cross-file primality algorithm surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primality;

impl InternalMpUint {
    /// Deterministic primality test for inputs below `2^64`.
    ///
    /// One limb uses the native Baillie-PSW variant. Wider inputs use
    /// Baillie-PSW with strong Lucas-Selfridge; both variants are exact below
    /// `2^64`, so every pointer width returns the same classification. Above
    /// `2^64`, passing is probable primality, not a proof.
    ///
    /// Returns `false` for 0 and 1.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        match self.limbs() {
            [] => false,
            [value] => Primality::is_prime_limb(*value),
            _ if self.is_even() => false,
            _ => Primality::baillie_psw(self),
        }
    }

    /// Returns `true` when `self` passes Miller-Rabin using a fixed prefix of
    /// prime bases.
    ///
    /// `k = 0` selects one base and values above 64 select at most 64 bases.
    /// Fixed bases make the result reproducible, but do not provide the
    /// independent-random-round error bound against adversarial inputs.
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        if self.is_zero() || self.is_one() {
            return false;
        }
        if self.is_even() {
            return self.cmp(&Self::from_limb(2)) == Ordering::Equal;
        }
        if self.to_u64().is_some() {
            return self.is_prime();
        }

        // self > 2^64 exceeds every screened prime, so a detected divisor is
        // a proper factor and rejects the input before exponentiation.
        if !Primality::trial_division(self) {
            return false;
        }

        // Write n-1 = d * 2^s with d odd. For odd self >= 3, (self - 1) >> s == self >> s.
        let s = Primality::trailing_zeros_odd_minus_one(self);
        let d = self.shr(s);

        // The preceding branches establish the domain's odd modulus > 1.
        let domain = MontgomeryDomain::new::<true>(self);

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

        // SAFETY: the active Limb slice occupies at most isize::MAX bytes;
        // adding one limb to its element count fits usize on 16-, 32-, and
        // 64-bit targets. Allocation still validates the requested byte capacity.
        let cap = unsafe { self.limbs().len().unchecked_add(1) };
        let mut temp_prod = Self::with_capacity(cap);
        let mut temp_rem = Self::with_capacity(cap);
        let mut mul_scratch = MontgomeryScratch::default();
        let one_mont =
            domain.transform_into_with_scratch(&Self::one(), &mut temp_prod, &mut mul_scratch);
        let n_minus_1_mont = self.sub(&one_mont);
        for &base in bases_to_check {
            let b = Self::from_u64(base);
            // The native path has returned, so self > u64::MAX > every base.
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

    /// Returns the smallest prime strictly greater than `self`.
    ///
    /// Candidates above `u64::MAX` use probable primality.
    #[must_use]
    pub fn next_prime(&self) -> Self {
        if self.cmp(&Self::from_limb(2)) == Ordering::Less {
            return Self::from_limb(2);
        }
        let step = if self.is_even() { 1 } else { 2 };
        let mut candidate = self.clone();
        candidate.add_assign(&Self::from_limb(step));
        Primality::search_prime::<false>(candidate)
    }

    /// Returns the largest prime strictly less than `self`.
    ///
    /// Returns `None` for inputs at most two. Candidates above `u64::MAX`
    /// use probable primality.
    #[must_use]
    pub fn prev_prime(&self) -> Option<Self> {
        if self.cmp(&Self::from_limb(2)) != Ordering::Greater {
            return None;
        }
        if self.cmp(&Self::from_limb(3)) == Ordering::Equal {
            return Some(Self::from_limb(2));
        }
        let step = if self.is_even() { 1 } else { 2 };
        let mut candidate = self.clone();
        candidate.sub_assign(&Self::from_limb(step));
        Some(Primality::search_prime::<true>(candidate))
    }
}
