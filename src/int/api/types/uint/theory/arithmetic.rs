//! Unsigned arithmetic-function APIs.

use super::{InternalMpUint, MpUint, Precision};

impl MpUint {
    /// Computes Euler's totient function $\phi(\text{self})$, counting integers $1 \le k \le \text{self}$
    /// coprime to `self`.
    ///
    /// Returns `None` if `self` is zero or if prime factorization cannot be completed.
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

    /// Computes the Jacobi symbol $\left(\frac{\text{self}}{\text{other}}\right)$.
    ///
    /// Returns `None` if `other` is zero or even, as the Jacobi symbol requires an odd modulus.
    #[must_use]
    pub fn jacobi_symbol(&self, other: &Self) -> Option<i8> {
        // Oddness establishes both Jacobi-domain conditions: nonzero and odd.
        if !other.value.is_odd() {
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
            value: InternalMpUint::factorial(n),
            precision,
        };
        result.assert_fits("factorial");
        result.debug_assert_valid();
        result
    }
}
