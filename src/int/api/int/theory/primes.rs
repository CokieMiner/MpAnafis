//! Signed primality APIs.

use super::{InternalMpInt, InternalMpUint, MpInt};

impl MpInt {
    /// Returns `true` if this value is positive and its absolute value is prime.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        self.is_positive() && self.value.abs.is_prime()
    }

    /// Returns `true` if this value is positive and its absolute value is probably prime using `k` rounds of Miller-Rabin.
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        self.is_positive() && self.value.abs.is_probably_prime(k)
    }

    /// Returns the smallest prime greater than or equal to this value.
    ///
    /// Negative inputs produce two when it fits. Returns `None` if the prime
    /// does not fit this value's bounded signed precision.
    #[must_use]
    pub fn next_prime(&self) -> Option<Self> {
        let next_abs = if self.is_negative() {
            InternalMpUint::from_u64(2)
        } else {
            self.value.abs.next_prime()
        };
        let value = InternalMpInt {
            abs: next_abs,
            is_positive: true,
        };
        if let Some(bits) = self.precision.significant_bits()
            && value.required_signed_bits_for_bounded_storage() > bits
        {
            return None;
        }
        let result = Self {
            value,
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }
}
