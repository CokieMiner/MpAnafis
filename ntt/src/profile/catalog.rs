//! Coefficient bounds for every admitted prefix of the prime catalog.

#![expect(
    unsafe_code,
    reason = "Constant profile construction uses the addressable literal catalog to prove indices and small integer arithmetic"
)]

use super::{MAX_PRIMES, PRIME_CATALOG};

/// Parameters of an exact multi-prime convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttProfile {
    /// Number of active primes in a nonempty catalog prefix.
    pub np: usize,
    /// Bits per nonnegative packed coefficient.
    pub chunk_bits: u32,
    /// Maximum admitted transform length for this profile.
    pub max_transform_len: usize,
}

/// Starting profiles; the planner narrows chunk widths to the actual coefficient bound.
pub const STANDARD_PROFILES: [NttProfile; MAX_PRIMES] = standard_profiles();

impl NttProfile {
    /// Tests the actual maximum number of products contributing to a coefficient.
    ///
    /// All catalog primes exceed `2^49`. The strict bound
    /// `terms*(2^chunk_bits-1)^2 < product(primes)` follows when
    /// `ceil(log2(terms)) + 2*chunk_bits <= 49*np`.
    #[must_use]
    pub fn admits(&self, transform_len: usize, terms: usize) -> bool {
        let Some(primes) = PRIME_CATALOG.get(..self.np) else {
            return false;
        };
        if primes.is_empty()
            || self.chunk_bits == 0
            || terms == 0
            || !transform_len.is_power_of_two()
            || transform_len < 4
            || terms > transform_len
            || transform_len > self.max_transform_len
        {
            return false;
        }
        let log_terms = usize::BITS.checked_sub(terms.saturating_sub(1).leading_zeros());
        let required = log_terms.and_then(|log| self.chunk_bits.checked_mul(2)?.checked_add(log));
        let capacity = u32::try_from(self.np)
            .ok()
            .and_then(|count| count.checked_mul(49));
        required
            .zip(capacity)
            .is_some_and(|(need, available)| need <= available)
            && primes
                .iter()
                .all(|prime| transform_len.trailing_zeros() <= prime.two_valuation())
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each constant profile width is checked against u32::MAX before narrowing; widening to u128 is exact"
)]
const fn standard_profiles() -> [NttProfile; MAX_PRIMES] {
    let mut profiles = [NttProfile {
        np: 0,
        chunk_bits: 0,
        max_transform_len: usize::MAX,
    }; MAX_PRIMES];
    let mut index = 0;
    // SAFETY: Prime occupies at least 32 bytes in an addressable catalog.
    // Thus 49*count fits usize, and each index is inside the equally sized
    // profile array. The assertion below admits the narrowed u32 width.
    unsafe {
        while index < MAX_PRIMES {
            let count = index.unchecked_add(1);
            let bits = count.unchecked_mul(49) >> 1;
            assert!(
                bits as u128 <= u32::MAX as u128,
                "catalog profile width must fit u32"
            );
            *profiles.as_mut_ptr().add(index) = NttProfile {
                np: count,
                chunk_bits: bits as u32,
                max_transform_len: usize::MAX,
            };
            index = count;
        }
    }
    profiles
}
