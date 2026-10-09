//! Signed number-theory methods built on unsigned magnitude algorithms.

use super::InternalMpInt;

impl InternalMpInt {
    /// Computes signed Bézout coefficients $(g, x, y)$ such that $g = \gcd(a, b) \ge 0$
    /// and $a \cdot x + b \cdot y = g$.
    ///
    /// Precondition: `other` must be non-zero.
    /// The shared kernel returns coefficient magnitudes and their opposing signs.
    /// Input signs are folded into those signs directly; zero remains positive.
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        debug_assert!(
            !other.abs.is_zero(),
            "signed extended GCD requires a non-zero second operand"
        );
        let result = self.abs.extended_gcd(&other.abs);
        let x = Self {
            is_positive: result.x_magnitude.is_zero() || result.x_is_positive == self.is_positive,
            abs: result.x_magnitude,
        };
        let y = Self {
            is_positive: result.y_magnitude.is_zero() || result.x_is_positive != other.is_positive,
            abs: result.y_magnitude,
        };

        (
            Self {
                abs: result.gcd,
                is_positive: true,
            },
            x,
            y,
        )
    }

    /// Computes the Jacobi symbol $\left(\frac{a}{m}\right)$ for odd modulus $m > 0$.
    ///
    /// The caller must establish a positive odd modulus. For negative `a`,
    /// multiplies the magnitude symbol by `(-1)^((m-1)/2)`, which is negative
    /// exactly when bit 1 of `m` is set.
    pub fn jacobi_symbol(&self, modulus: &Self) -> i8 {
        debug_assert!(
            modulus.is_positive && !modulus.abs.is_zero() && modulus.abs.is_odd(),
            "signed Jacobi requires a positive odd modulus"
        );
        let symbol = self.abs.jacobi_symbol(&modulus.abs);
        if !self.is_positive && modulus.abs.get_bit(1) {
            symbol.wrapping_neg()
        } else {
            symbol
        }
    }
}
