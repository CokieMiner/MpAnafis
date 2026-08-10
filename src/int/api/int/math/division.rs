//! Signed division and divisibility APIs.

use super::{ArbiInt, InternalArbiInt, Precision};

impl ArbiInt {
    /// Returns the quotient and remainder of truncating division.
    ///
    /// Returns `None` when `rhs` is zero or bounded `MIN / -1` would overflow.
    #[must_use]
    pub fn div_rem(&self, rhs: &Self) -> Option<(Self, Self)> {
        let p = combined_division_precision(self, rhs)?;
        let (quotient_value, remainder_value) = self.value.div_rem(&rhs.value);
        let quotient = Self {
            value: quotient_value,
            precision: p,
        };
        quotient.debug_assert_valid();
        let remainder = Self {
            value: remainder_value,
            precision: p,
        };
        remainder.debug_assert_valid();
        Some((quotient, remainder))
    }

    /// Returns `true` if `self` is divisible by `other` (i.e., `self % other == 0`).
    ///
    /// By convention, zero is divisible by zero; no nonzero value is divisible
    /// by zero.
    #[must_use]
    pub fn is_divisible_by(&self, other: &Self) -> bool {
        self.value.abs.is_divisible_by(&other.value.abs)
    }

    /// Returns `true` if `self` divides `other` (i.e., `other % self == 0`).
    ///
    /// By convention, zero divides zero but no other value.
    #[must_use]
    pub fn is_divisor_of(&self, other: &Self) -> bool {
        other.is_divisible_by(self)
    }

    /// Truncating division (identical to `/`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_trunc(&self, rhs: &Self) -> Self {
        let precision =
            combined_division_precision(self, rhs).expect("division by zero or bounded overflow");
        let quotient = Self {
            value: self.value.div(&rhs.value),
            precision,
        };
        quotient.debug_assert_valid();
        quotient
    }

    /// Checked truncating division.
    #[must_use]
    pub fn checked_div_trunc(&self, rhs: &Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Truncating remainder (identical to `%`).
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn rem_trunc(&self, rhs: &Self) -> Self {
        checked_truncating_remainder(self, rhs).expect("division by zero or bounded overflow")
    }

    /// Checked truncating remainder.
    #[must_use]
    pub fn checked_rem_trunc(&self, rhs: &Self) -> Option<Self> {
        checked_truncating_remainder(self, rhs)
    }

    /// Returns the quotient and remainder of Euclidean division.
    #[must_use]
    pub fn div_rem_euclid(&self, rhs: &Self) -> Option<(Self, Self)> {
        let (mut q, mut r) = self.div_rem(rhs)?;
        if r.is_negative() {
            if rhs.is_positive() {
                q.value = q.value.sub(&InternalArbiInt::one());
                r.value = r.value.add(&rhs.value);
            } else {
                q.value = q.value.add(&InternalArbiInt::one());
                r.value = r.value.sub(&rhs.value);
            }
        }
        Some((q, r))
    }

    /// Euclidean division.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_euclid(&self, rhs: &Self) -> Self {
        self.div_rem_euclid(rhs)
            .map(|(q, _)| q)
            .expect("division by zero or bounded overflow")
    }

    /// Checked Euclidean division.
    #[must_use]
    pub fn checked_div_euclid(&self, rhs: &Self) -> Option<Self> {
        self.div_rem_euclid(rhs).map(|(q, _)| q)
    }

    /// Euclidean remainder.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn rem_euclid(&self, rhs: &Self) -> Self {
        let remainder =
            checked_truncating_remainder(self, rhs).expect("division by zero or bounded overflow");
        euclidean_remainder(remainder, rhs)
    }

    /// Checked Euclidean remainder.
    #[must_use]
    pub fn checked_rem_euclid(&self, rhs: &Self) -> Option<Self> {
        Some(euclidean_remainder(
            checked_truncating_remainder(self, rhs)?,
            rhs,
        ))
    }

    /// Returns the quotient and remainder of floor division.
    #[must_use]
    pub fn div_rem_floor(&self, rhs: &Self) -> Option<(Self, Self)> {
        let (mut q, mut r) = self.div_rem(rhs)?;
        if (self.is_negative() != rhs.is_negative()) && !r.is_zero() {
            q.value = q.value.sub(&InternalArbiInt::one());
            r.value = r.value.add(&rhs.value);
        }
        Some((q, r))
    }

    /// Floor division. Rounds quotient toward negative infinity.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_floor(&self, rhs: &Self) -> Self {
        self.div_rem_floor(rhs)
            .map(|(q, _)| q)
            .expect("division by zero or bounded overflow")
    }

    /// Checked floor division.
    #[must_use]
    pub fn checked_div_floor(&self, rhs: &Self) -> Option<Self> {
        self.div_rem_floor(rhs).map(|(q, _)| q)
    }

    /// Floor modulus.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn mod_floor(&self, rhs: &Self) -> Self {
        let remainder =
            checked_truncating_remainder(self, rhs).expect("division by zero or bounded overflow");
        floor_remainder(remainder, self, rhs)
    }

    /// Checked floor modulus.
    #[must_use]
    pub fn checked_mod_floor(&self, rhs: &Self) -> Option<Self> {
        Some(floor_remainder(
            checked_truncating_remainder(self, rhs)?,
            self,
            rhs,
        ))
    }

    /// Ceiling division. Rounds quotient toward positive infinity.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_ceil(&self, rhs: &Self) -> Self {
        let (mut q, r) = self
            .div_rem(rhs)
            .expect("division by zero or bounded overflow");
        if !r.is_zero() && (self.is_negative() == rhs.is_negative()) {
            q.value = q.value.add(&InternalArbiInt::one());
        }
        q
    }

    /// Checked ceiling division.
    #[must_use]
    pub fn checked_div_ceil(&self, rhs: &Self) -> Option<Self> {
        let (mut q, r) = self.div_rem(rhs)?;
        if !r.is_zero() && (self.is_negative() == rhs.is_negative()) {
            // After rejecting bounded `MIN / -1`, rounding a same-sign
            // quotient upward cannot exceed the dividend's signed range.
            // Mutating the internal value retains the combined precision;
            // adding public `Self::one()` would incorrectly promote it to
            // unlimited precision.
            q.value = q.value.add(&InternalArbiInt::one());
        }
        q.debug_assert_valid();
        Some(q)
    }
}

fn combined_division_precision(lhs: &ArbiInt, rhs: &ArbiInt) -> Option<Precision> {
    if rhs.value.abs.is_zero() {
        return None;
    }
    let precision = lhs.precision.combine_for_binary_op(rhs.precision);
    if let Some(bits) = precision.significant_bits()
        && lhs.value.bounded_division_overflows(&rhs.value, bits)
    {
        return None;
    }
    Some(precision)
}

fn checked_truncating_remainder(lhs: &ArbiInt, rhs: &ArbiInt) -> Option<ArbiInt> {
    let precision = combined_division_precision(lhs, rhs)?;
    let remainder = ArbiInt {
        value: lhs.value.rem(&rhs.value),
        precision,
    };
    remainder.debug_assert_valid();
    Some(remainder)
}

fn euclidean_remainder(mut remainder: ArbiInt, rhs: &ArbiInt) -> ArbiInt {
    if remainder.is_negative() {
        // The truncating remainder has sign(lhs) and |r| < |rhs|. Adding
        // |rhs| therefore yields the unique representative in [0, |rhs|).
        remainder.value = if rhs.is_positive() {
            remainder.value.add(&rhs.value)
        } else {
            remainder.value.sub(&rhs.value)
        };
    }
    remainder.debug_assert_valid();
    remainder
}

fn floor_remainder(mut remainder: ArbiInt, lhs: &ArbiInt, rhs: &ArbiInt) -> ArbiInt {
    if !remainder.is_zero() && lhs.is_negative() != rhs.is_negative() {
        // Truncation and floor differ by one quotient unit exactly when the
        // signs differ and r != 0, so r_floor = r_trunc + rhs.
        remainder.value = remainder.value.add(&rhs.value);
    }
    remainder.debug_assert_valid();
    remainder
}
