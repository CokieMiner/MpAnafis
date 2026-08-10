//! Signed greatest-common-divisor APIs.

use super::{ArbiInt, InternalArbiInt};

impl ArbiInt {
    /// Returns the greatest common divisor of the absolute values of `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the non-negative gcd cannot be represented in the resolved
    /// bounded signed precision, as can happen for `gcd(MIN, 0)`.
    #[must_use]
    #[track_caller]
    pub fn gcd(&self, other: &Self) -> Self {
        let g = self.value.abs.gcd(&other.value.abs);
        let p = self.precision.combine_for_binary_op(other.precision);
        let gcd = Self {
            value: InternalArbiInt {
                abs: g,
                is_positive: true,
            },
            precision: p,
        };
        gcd.assert_fits("gcd");
        gcd.debug_assert_valid();
        gcd
    }

    /// Returns the gcd and lcm of the absolute values, or `None` if either
    /// non-negative result cannot fit the resolved bounded signed precision.
    #[must_use]
    pub fn gcd_lcm(&self, other: &Self) -> Option<(Self, Self)> {
        let (g, l) = self.value.abs.gcd_lcm(&other.value.abs);
        let p = self.precision.combine_for_binary_op(other.precision);
        let gcd = Self {
            value: InternalArbiInt {
                abs: g,
                is_positive: true,
            },
            precision: p,
        };
        let lcm = Self {
            value: InternalArbiInt {
                abs: l,
                is_positive: true,
            },
            precision: p,
        };
        if let Some(bits) = p.significant_bits() {
            let gcd_bits = gcd.value.required_signed_bits_for_bounded_storage();
            let lcm_bits = lcm.value.required_signed_bits_for_bounded_storage();
            if gcd_bits > bits || lcm_bits > bits {
                return None;
            }
        }
        gcd.debug_assert_valid();
        lcm.debug_assert_valid();
        Some((gcd, lcm))
    }

    /// Returns the least common multiple of the absolute values, or `None` if
    /// the non-negative result cannot fit the resolved bounded signed precision.
    #[must_use]
    pub fn lcm(&self, other: &Self) -> Option<Self> {
        let p = self.precision.combine_for_binary_op(other.precision);
        let result = Self {
            value: InternalArbiInt {
                abs: self.value.abs.lcm(&other.value.abs),
                is_positive: true,
            },
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && result.value.required_signed_bits_for_bounded_storage() > bits
        {
            return None;
        }
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `true` if the absolute values are coprime.
    #[must_use]
    pub fn is_coprime(&self, other: &Self) -> bool {
        self.value.abs.is_coprime(&other.value.abs)
    }

    /// Computes the extended GCD, returning `(gcd, x, y)` such that
    /// `self * x + other * y = gcd`.
    ///
    /// Returns `None` when `other` is zero or when a result cannot fit the
    /// resolved bounded signed precision.
    #[must_use]
    pub fn extended_gcd(&self, other: &Self) -> Option<(Self, Self, Self)> {
        if other.value.abs.is_zero() {
            return None;
        }
        let p = self.precision.combine_for_binary_op(other.precision);
        let (gcd_value, x_value, y_value) = self.value.extended_gcd(&other.value);
        let gcd = Self {
            value: gcd_value,
            precision: p,
        };
        let x = Self {
            value: x_value,
            precision: p,
        };
        let y = Self {
            value: y_value,
            precision: p,
        };
        if let Some(bits) = p.significant_bits()
            && [
                gcd.value.required_signed_bits_for_bounded_storage(),
                x.value.required_signed_bits_for_bounded_storage(),
                y.value.required_signed_bits_for_bounded_storage(),
            ]
            .into_iter()
            .any(|required| required > bits)
        {
            return None;
        }
        gcd.debug_assert_valid();
        x.debug_assert_valid();
        y.debug_assert_valid();
        Some((gcd, x, y))
    }
}
