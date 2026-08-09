//! Validated signed division kernels implemented on `InternalMpInt`.

#![allow(
    clippy::same_name_method,
    reason = "InternalMpInt inherent division deliberately mirrors the corresponding operator traits."
)]

use super::InternalMpInt;

impl InternalMpInt {
    /// Returns whether truncating division would overflow a signed `bits`-bit
    /// destination.
    ///
    /// The dividend must already fit the caller-validated non-zero destination
    /// width; the divisor may have any width. Integer division never increases
    /// the dividend magnitude except when negating the unique asymmetric
    /// endpoint: `MIN_bits / -1 = 2^(bits - 1)`, one above `MAX_bits`.
    #[inline]
    #[must_use]
    pub fn bounded_division_overflows(&self, rhs: &Self, bits: usize) -> bool {
        debug_assert!(bits > 0, "signed division width must be non-zero");
        self.is_signed_min_for_width(bits) && !rhs.is_positive && rhs.abs.is_one()
    }

    /// Computes truncating quotient and remainder by a caller-validated
    /// non-zero signed divisor.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed division requires a non-zero divisor"
        );
        let (quotient_abs, remainder_abs) = self.abs.div_rem(&divisor.abs);

        // Magnitude division gives |a| = |q| |b| + |r| with |r| < |b|.
        // Truncation toward zero assigns sign(a) xor sign(b) to q and sign(a)
        // to r; normalization makes either exact-zero result positive.
        let quotient = Self {
            abs: quotient_abs,
            is_positive: self.is_positive == divisor.is_positive,
        }
        .normalized();
        let remainder = Self {
            abs: remainder_abs,
            is_positive: self.is_positive,
        }
        .normalized();
        (quotient, remainder)
    }

    /// Divides by a caller-validated non-zero signed divisor.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn div(&self, divisor: &Self) -> Self {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed division requires a non-zero divisor"
        );
        let quotient = self.abs.div(&divisor.abs);
        Self {
            abs: quotient,
            is_positive: self.is_positive == divisor.is_positive,
        }
        .normalized()
    }

    /// Computes remainder by a caller-validated non-zero signed divisor.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn rem(&self, divisor: &Self) -> Self {
        debug_assert!(
            !divisor.abs.is_zero(),
            "signed remainder requires a non-zero divisor"
        );
        let remainder = self.abs.rem(&divisor.abs);
        Self {
            abs: remainder,
            is_positive: self.is_positive,
        }
        .normalized()
    }

    /// Divides this value in place by a caller-validated non-zero divisor.
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

    /// Replaces this value with remainder by a caller-validated non-zero divisor.
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
