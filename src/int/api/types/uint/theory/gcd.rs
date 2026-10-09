//! Unsigned greatest-common-divisor APIs.

use super::MpUint;

impl MpUint {
    /// Returns the greatest common divisor of `self` and `other`.
    #[must_use]
    pub fn gcd(&self, other: &Self) -> Self {
        let g = self.value.gcd(&other.value);
        let p = self.precision.combine_for_binary_op(other.precision);
        let gcd = Self {
            value: g,
            precision: p,
        };
        gcd.debug_assert_valid();
        gcd
    }

    /// Returns the greatest common divisor and least common multiple, or
    /// `None` if the lcm cannot be computed or fit the resolved bounded
    /// precision.
    #[must_use]
    pub fn gcd_lcm(&self, other: &Self) -> Option<(Self, Self)> {
        let (g, l) = self.value.gcd_lcm(&other.value);
        let p = self.precision.combine_for_binary_op(other.precision);
        let gcd = Self {
            value: g,
            precision: p,
        };
        gcd.debug_assert_valid();
        let lcm = Self {
            value: l,
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && lcm.value.required_unsigned_bits_for_bounded_storage() > bits
        {
            return None;
        }
        lcm.debug_assert_valid();
        Some((gcd, lcm))
    }

    /// Returns the least common multiple, or `None` if it cannot be computed
    /// or fit the resolved bounded precision.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Option<Self> {
        let p = self.precision.combine_for_binary_op(other.precision);
        let result = Self {
            value: self.value.lcm(&other.value),
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && result.value.required_unsigned_bits_for_bounded_storage() > bits
        {
            return None;
        }
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `true` if `self` and `other` are coprime (share no common divisors other than 1).
    #[must_use]
    pub fn is_coprime(&self, other: &Self) -> bool {
        self.value.is_coprime(&other.value)
    }

    /// Computes the extended GCD and unsigned Bezout-coefficient residues.
    ///
    /// For nonzero operands, the returned `(gcd, x, y)` satisfies
    /// `self*x = gcd (mod other)` and `other*y = gcd (mod self)`. Use the
    /// signed integer API when ordinary signed Bezout coefficients are needed.
    /// Returns `None` when `other` is zero.
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> Option<(Self, Self, Self)> {
        if other.value.is_zero() {
            return None;
        }
        let p = self.precision.combine_for_binary_op(other.precision);
        let mut result = self.value.extended_gcd(&other.value);
        // The shared kernel returns absolute coefficients. Only the unsigned
        // API requires the negative coefficient's residue representative.
        // Exact fallbacks retain the public interval for every valid input.
        if !result.x_magnitude.is_zero() && !result.y_magnitude.is_zero() {
            if result.x_magnitude >= other.value {
                result.x_magnitude.rem_assign(&other.value);
            }
            if result.y_magnitude >= self.value {
                result.y_magnitude.rem_assign(&self.value);
            }
            if result.x_is_positive && !result.y_magnitude.is_zero() {
                result.y_magnitude.negate_reduced_modulo(&self.value);
            } else if !result.x_is_positive && !result.x_magnitude.is_zero() {
                result.x_magnitude.negate_reduced_modulo(&other.value);
            }
        }
        let gcd = Self {
            value: result.gcd,
            precision: p,
        };
        gcd.debug_assert_valid();
        let x_val = Self {
            value: result.x_magnitude,
            precision: p,
        };
        x_val.debug_assert_valid();
        let y_val = Self {
            value: result.y_magnitude,
            precision: p,
        };
        y_val.debug_assert_valid();
        Some((gcd, x_val, y_val))
    }
}
