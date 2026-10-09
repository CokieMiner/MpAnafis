//! Signed division and divisibility APIs.

use super::{InternalMpInt, MpInt, Precision};

impl MpInt {
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
            // A negative dividend rounds away from zero: either q <= 0
            // is decremented or q >= 0 is incremented. Both increase |q|.
            q.value.abs.increment();
            if rhs.is_positive() {
                q.value.is_positive = false;
                r.value.add_assign(&rhs.value);
            } else {
                r.value.sub_assign(&rhs.value);
            }
        }
        q.debug_assert_valid();
        r.debug_assert_valid();
        Some((q, r))
    }

    /// Euclidean division.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_euclid(&self, rhs: &Self) -> Self {
        self.checked_div_euclid(rhs)
            .expect("division by zero or bounded overflow")
    }

    /// Checked Euclidean division.
    #[must_use]
    pub fn checked_div_euclid(&self, rhs: &Self) -> Option<Self> {
        checked_rounded_quotient(self, rhs, self.is_negative())
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
            // The truncating quotient is non-positive, so q - 1 has
            // magnitude |q| + 1 and a strictly negative sign, even for q = 0.
            q.value.abs.increment();
            q.value.is_positive = false;
            r.value.add_assign(&rhs.value);
        }
        q.debug_assert_valid();
        r.debug_assert_valid();
        Some((q, r))
    }

    /// Floor division. Rounds quotient toward negative infinity.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or on bounded overflow (`MIN / -1`).
    #[must_use]
    pub fn div_floor(&self, rhs: &Self) -> Self {
        self.checked_div_floor(rhs)
            .expect("division by zero or bounded overflow")
    }

    /// Checked floor division.
    #[must_use]
    pub fn checked_div_floor(&self, rhs: &Self) -> Option<Self> {
        checked_rounded_quotient(self, rhs, self.is_negative() != rhs.is_negative())
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
        self.checked_div_ceil(rhs)
            .expect("division by zero or bounded overflow")
    }

    /// Checked ceiling division.
    #[must_use]
    pub fn checked_div_ceil(&self, rhs: &Self) -> Option<Self> {
        checked_rounded_quotient(self, rhs, self.is_negative() == rhs.is_negative())
    }
}

fn checked_rounded_quotient(lhs: &MpInt, rhs: &MpInt, away_from_zero: bool) -> Option<MpInt> {
    let precision = combined_division_precision(lhs, rhs)?;
    // Rounding away from zero takes ceil(|lhs|/|rhs|); all other cases
    // take floor(|lhs|/|rhs|) and need no remainder. Rejecting MIN/-1
    // makes both rounded magnitudes representable with the combined width.
    let abs = if away_from_zero {
        lhs.value.abs.div_ceil(&rhs.value.abs)
    } else {
        lhs.value.abs.div(&rhs.value.abs)
    };
    let result = MpInt {
        value: InternalMpInt {
            is_positive: abs.is_zero() || lhs.value.is_positive == rhs.value.is_positive,
            abs,
        },
        precision,
    };
    result.debug_assert_valid();
    Some(result)
}

fn checked_truncating_remainder(lhs: &MpInt, rhs: &MpInt) -> Option<MpInt> {
    let precision = combined_division_precision(lhs, rhs)?;
    let remainder = MpInt {
        value: lhs.value.rem(&rhs.value),
        precision,
    };
    remainder.debug_assert_valid();
    Some(remainder)
}

fn euclidean_remainder(mut remainder: MpInt, rhs: &MpInt) -> MpInt {
    if remainder.is_negative() {
        // The truncating remainder has sign(lhs) and |r| < |rhs|. Adding
        // |rhs| therefore yields the unique representative in [0, |rhs|).
        if rhs.is_positive() {
            remainder.value.add_assign(&rhs.value);
        } else {
            remainder.value.sub_assign(&rhs.value);
        }
    }
    remainder.debug_assert_valid();
    remainder
}

fn floor_remainder(mut remainder: MpInt, lhs: &MpInt, rhs: &MpInt) -> MpInt {
    if !remainder.is_zero() && lhs.is_negative() != rhs.is_negative() {
        // Truncation and floor differ by one quotient unit exactly when the
        // signs differ and r != 0, so r_floor = r_trunc + rhs.
        remainder.value.add_assign(&rhs.value);
    }
    remainder.debug_assert_valid();
    remainder
}

fn combined_division_precision(lhs: &MpInt, rhs: &MpInt) -> Option<Precision> {
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
