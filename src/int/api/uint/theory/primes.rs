//! Unsigned primality APIs.

use super::MpUint;

impl MpUint {
    /// Returns `true` if the value is prime.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        self.value.is_prime()
    }

    /// Returns `true` if the value is probably prime using `k` rounds of Miller-Rabin.
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        self.value.is_probably_prime(k)
    }

    /// Returns the smallest prime greater than or equal to this value.
    ///
    /// Returns `None` if the prime does not fit this value's bounded precision.
    #[must_use]
    pub fn next_prime(&self) -> Option<Self> {
        let value = self.value.next_prime();
        if let Some(bits) = self.precision.significant_bits()
            && value.required_unsigned_bits_for_bounded_storage() > bits
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
