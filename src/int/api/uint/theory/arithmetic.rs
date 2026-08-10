//! Unsigned arithmetic-function APIs.

use super::{ArbiUint, InternalArbiUint, Precision};

impl ArbiUint {
    /// Returns the Euler totient function value of this number, or `None` if it cannot be computed.
    #[must_use]
    pub fn euler_phi(&self) -> Option<Self> {
        self.value.euler_phi().map(|v| {
            let result = Self {
                value: v,
                precision: self.precision,
            };
            result.debug_assert_valid();
            result
        })
    }

    /// Returns the Jacobi symbol of this value with respect to the modulus `other`, or `None` if it is undefined.
    #[must_use]
    pub fn jacobi_symbol(&self, other: &Self) -> Option<i8> {
        if other.value.is_zero() || other.value.is_even() {
            return None;
        }
        Some(self.value.jacobi_symbol(&other.value))
    }

    /// Computes the factorial of `n` (`n!`).
    ///
    /// # Panics
    ///
    /// Panics if the result exceeds `precision` when it is bounded.
    #[must_use]
    #[track_caller]
    pub fn factorial(n: u32, precision: Precision) -> Self {
        let result = Self {
            value: InternalArbiUint::factorial(n),
            precision,
        };
        result.assert_fits("factorial");
        result.debug_assert_valid();
        result
    }
}
