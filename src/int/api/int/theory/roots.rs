//! Signed root APIs.

use super::{ArbiInt, InternalArbiInt};

impl ArbiInt {
    /// Returns the floor square root of the integer, or `None` if the integer is negative.
    #[must_use]
    pub fn checked_isqrt(&self) -> Option<Self> {
        if self.is_negative() {
            return None;
        }
        let result = Self {
            value: InternalArbiInt {
                abs: self.value.abs.isqrt(),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns the floor square root and remainder of the absolute value.
    #[must_use]
    pub fn sqrt_rem(&self) -> Option<(Self, Self)> {
        let (s, r) = self.value.abs.sqrt_rem();
        let sq = Self {
            value: InternalArbiInt {
                abs: s,
                is_positive: true,
            },
            precision: self.precision,
        };
        sq.debug_assert_valid();
        let rem = Self {
            value: InternalArbiInt {
                abs: r,
                is_positive: true,
            },
            precision: self.precision,
        };
        rem.debug_assert_valid();
        Some((sq, rem))
    }

    /// Returns the floor n-th root of the absolute value.
    #[must_use]
    pub fn nth_root(&self, n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let result = Self {
            value: InternalArbiInt {
                abs: self.value.abs.nth_root(n),
                is_positive: true,
            },
            precision: self.precision,
        };
        result.debug_assert_valid();
        Some(result)
    }

    /// Returns `true` if the absolute value is a perfect square.
    #[must_use]
    pub fn is_perfect_square(&self) -> bool {
        self.value.abs.is_perfect_square()
    }
}
