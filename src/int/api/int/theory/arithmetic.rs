//! Signed arithmetic-function APIs.

use super::{ArbiInt, InternalArbiInt, InternalArbiUint, Precision};

impl ArbiInt {
    /// Returns the Euler totient of this positive integer.
    ///
    /// Returns `None` when the value is zero or negative.
    #[must_use]
    pub fn euler_phi(&self) -> Option<Self> {
        if !self.is_positive() {
            return None;
        }
        self.value.abs.euler_phi().map(|v| {
            let result = Self {
                value: InternalArbiInt {
                    abs: v,
                    is_positive: true,
                },
                precision: self.precision,
            };
            result.debug_assert_valid();
            result
        })
    }

    /// Returns the Jacobi symbol of this signed numerator modulo `other`.
    ///
    /// Returns `None` unless `other` is a positive odd integer.
    #[must_use]
    pub fn jacobi_symbol(&self, other: &Self) -> Option<i8> {
        if !other.value.is_positive || other.value.abs.is_zero() || other.value.abs.is_even() {
            return None;
        }
        Some(self.value.jacobi_symbol(&other.value))
    }

    /// Computes the factorial of `n`.
    ///
    /// # Panics
    ///
    /// Panics if the result exceeds `precision` when it is bounded.
    #[must_use]
    #[track_caller]
    pub fn factorial(n: u32, precision: Precision) -> Self {
        let result = Self {
            value: InternalArbiInt {
                abs: InternalArbiUint::factorial(n),
                is_positive: true,
            },
            precision,
        };
        result.assert_fits("factorial");
        result.debug_assert_valid();
        result
    }
}
