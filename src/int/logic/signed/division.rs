//! Truncating signed division for a caller-validated nonzero divisor.

use super::InternalMpInt;

impl InternalMpInt {
    /// Returns whether truncating division $a / b$ overflows a signed `bits`-bit destination.
    ///
    /// The dividend must fit the destination bit width. Division never increases magnitude
    /// except when negating the asymmetric two's-complement boundary:
    /// $\operatorname{MIN}_w / -1 = (-2^{w-1}) / -1 = 2^{w-1}$, which exceeds $\operatorname{MAX}_w = 2^{w-1} - 1$.
    #[inline]
    #[must_use]
    pub fn bounded_division_overflows(&self, rhs: &Self, bits: usize) -> bool {
        debug_assert!(bits > 0, "signed division width must be non-zero");
        self.is_signed_min_for_width(bits) && !rhs.is_positive && rhs.abs.is_one()
    }

    /// Computes truncating quotient $q$ and remainder $r$ such that $a = q \cdot b + r$ with $|r| < |b|$.
    ///
    /// Rounds toward zero. A nonzero quotient is positive exactly when the
    /// operand signs agree; a nonzero remainder has the dividend's sign.
    /// The caller must establish that the divisor is nonzero.
    #[inline]
    #[must_use]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed division requires a non-zero divisor"
        );
        let (quotient_abs, remainder_abs) = self.abs.div_rem(&divisor.abs);

        let quotient_sign = self.is_positive == divisor.is_positive;
        let quotient = Self {
            is_positive: quotient_abs.is_zero() || quotient_sign,
            abs: quotient_abs,
        };
        let remainder = Self {
            is_positive: remainder_abs.is_zero() || self.is_positive,
            abs: remainder_abs,
        };
        (quotient, remainder)
    }

    /// Computes truncating quotient $q = \operatorname{trunc}(a / b)$.
    ///
    /// The caller must establish that the divisor is nonzero.
    #[inline]
    #[must_use]
    pub fn div(&self, divisor: &Self) -> Self {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed division requires a non-zero divisor"
        );
        let quotient_abs = self.abs.div(&divisor.abs);
        let quotient_sign = self.is_positive == divisor.is_positive;
        Self {
            is_positive: quotient_abs.is_zero() || quotient_sign,
            abs: quotient_abs,
        }
    }

    /// Computes truncating remainder $r = a - q \cdot b$.
    ///
    /// The caller must establish that the divisor is nonzero.
    #[inline]
    #[must_use]
    pub fn rem(&self, divisor: &Self) -> Self {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed remainder requires a non-zero divisor"
        );
        let remainder_abs = self.abs.rem(&divisor.abs);
        Self {
            is_positive: remainder_abs.is_zero() || self.is_positive,
            abs: remainder_abs,
        }
    }

    /// Computes truncating division in place ($a \leftarrow \operatorname{trunc}(a / b)$).
    ///
    /// The caller must establish that the divisor is nonzero.
    #[inline]
    pub fn div_assign(&mut self, divisor: &Self) {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed division requires a non-zero divisor"
        );
        self.abs.div_assign(&divisor.abs);
        self.is_positive = self.is_positive == divisor.is_positive;
        if self.abs.is_zero() {
            self.is_positive = true;
        }
    }

    /// Computes truncating remainder in place ($a \leftarrow a \pmod b$).
    ///
    /// The caller must establish that the divisor is nonzero.
    #[inline]
    pub fn rem_assign(&mut self, divisor: &Self) {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed remainder requires a non-zero divisor"
        );
        self.abs.rem_assign(&divisor.abs);
        if self.abs.is_zero() {
            self.is_positive = true;
        }
    }
}
