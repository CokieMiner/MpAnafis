//! Signed primality APIs.

use super::{InternalMpInt, InternalMpUint, MpInt};

impl MpInt {
    /// Returns `true` if this value is positive and its magnitude is prime for
    /// values that fit in `u64`.
    ///
    /// Larger magnitudes use Baillie-PSW (base-2 Miller-Rabin and strong
    /// Lucas-Selfridge); `true` then indicates probable primality, not a proof.
    #[must_use]
    pub fn is_prime(&self) -> bool {
        !self.is_negative() && self.value.abs.is_prime()
    }

    /// Returns `true` if this value is positive and its magnitude passes a
    /// fixed-base Miller-Rabin test.
    ///
    /// `k = 0` selects one base and values above 64 select at most 64 bases.
    #[must_use]
    pub fn is_probably_prime(&self, k: u32) -> bool {
        !self.is_negative() && self.value.abs.is_probably_prime(k)
    }

    /// Returns the smallest prime strictly greater than this value.
    ///
    /// Negative inputs produce two when it fits. Returns `None` if the prime
    /// does not fit this value's bounded signed precision.
    /// Candidates above `u64::MAX` use probable primality.
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

    /// Returns the largest positive prime strictly less than this value.
    ///
    /// Returns `None` for values at most two. The result preserves this
    /// value's precision. Candidates above `u64::MAX` use probable primality.
    #[must_use]
    pub fn prev_prime(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let result = Self {
            value: InternalMpInt {
                abs: self.value.abs.prev_prime()?,
                is_positive: true,
            },
            precision: self.precision,
        };
        // A smaller positive result fits the signed precision of positive self.
        result.debug_assert_valid();
        Some(result)
    }
}
