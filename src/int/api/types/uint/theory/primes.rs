//! Unsigned primality APIs.

use super::MpUint;

impl MpUint {
    /// Returns `true` if the value is prime for values that fit in `u64`.
    ///
    /// Larger values use Baillie-PSW (base-2 Miller-Rabin and strong
    /// Lucas-Selfridge); `true` then indicates probable primality, not a proof.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        self.value.is_prime()
    }

    /// Returns `true` if the value passes a fixed-base Miller-Rabin test.
    ///
    /// `k = 0` selects one base and values above 64 select at most 64 bases.
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        self.value.is_probably_prime(k)
    }

    /// Returns the smallest prime strictly greater than this value.
    ///
    /// Returns `None` if the prime does not fit this value's bounded precision.
    /// Candidates above `u64::MAX` use probable primality.
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

    /// Returns the largest prime strictly less than this value.
    ///
    /// Returns `None` for values at most two. The result preserves this
    /// value's precision. Candidates above `u64::MAX` use probable primality.
    #[must_use]
    pub fn prev_prime(&self) -> Option<Self> {
        let result = Self {
            value: self.value.prev_prime()?,
            precision: self.precision,
        };
        // A smaller nonnegative result fits every precision that admits self.
        result.debug_assert_valid();
        Some(result)
    }
}
