//! Signed number-theory methods built on unsigned magnitude algorithms.

use core::cmp::Ordering;

use super::InternalMpInt;

impl InternalMpInt {
    /// Computes signed Bezout coefficients for a non-zero second operand.
    ///
    /// The unsigned kernel returns coefficient residues. Complementing one
    /// residue recovers an ordinary signed Bezout pair.
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        debug_assert!(
            !other.abs.is_zero(),
            "signed extended GCD requires a non-zero second operand"
        );
        let (gcd_abs, x_residue, y_residue) = self.abs.extended_gcd(&other.abs);
        let (mut x, mut y) = if x_residue.is_zero() || y_residue.is_zero() {
            (
                Self {
                    abs: x_residue,
                    is_positive: true,
                },
                Self {
                    abs: y_residue,
                    is_positive: true,
                },
            )
        } else if self.abs.mul(&x_residue).cmp(&other.abs.mul(&y_residue)) == Ordering::Less {
            let y_magnitude = self.abs.sub(&y_residue);
            (
                Self {
                    abs: x_residue,
                    is_positive: true,
                },
                Self {
                    abs: y_magnitude,
                    is_positive: false,
                },
            )
        } else {
            let x_magnitude = other.abs.sub(&x_residue);
            (
                Self {
                    abs: x_magnitude,
                    is_positive: false,
                },
                Self {
                    abs: y_residue,
                    is_positive: true,
                },
            )
        };

        if !self.is_positive && !x.abs.is_zero() {
            x.is_positive = !x.is_positive;
        }
        if !other.is_positive && !y.abs.is_zero() {
            y.is_positive = !y.is_positive;
        }

        (
            Self {
                abs: gcd_abs,
                is_positive: true,
            },
            x,
            y,
        )
    }

    /// Computes the Jacobi symbol for a caller-validated positive odd modulus.
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
