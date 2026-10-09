//! Signed two's-complement wrapping implemented on `InternalMpInt`.

use super::InternalMpInt;

impl InternalMpInt {
    /// Wraps this signed integer into the $w$-bit two's-complement interval
    /// $[-2^{w-1}, 2^{w-1} - 1]$ modulo $2^w$.
    ///
    /// Precondition: `bits > 0`.
    /// Zero residues have a positive sign.
    #[inline]
    #[must_use]
    pub fn apply_wrapping(self, bits: usize) -> Self {
        self.apply_wrapping_with_overflow(bits).0
    }

    /// Reduces modulo `2^bits`, interpreting the sign bit and reporting whether
    /// the original value lies outside `[-2^(bits-1), 2^(bits-1)-1]`.
    ///
    /// The caller must establish `bits > 0`.
    #[inline]
    #[must_use]
    pub fn apply_wrapping_with_overflow(self, bits: usize) -> (Self, bool) {
        debug_assert!(bits > 0, "signed wrapping width must be non-zero");
        let significant_bits = self.abs.significant_bits();
        // Both signs fit strictly below the sign-bit width. Only the negative
        // endpoint at exactly bits significant bits needs a power-of-two test.
        if significant_bits < bits
            || (!self.is_positive && significant_bits == bits && self.abs.is_power_of_two())
        {
            return (self, false);
        }

        if significant_bits == bits {
            // The magnitude is already in [2^(bits-1), 2^bits), so neither
            // truncation nor another sign-bit test is needed. The negative
            // endpoint returned above; every remaining value wraps by replacing
            // its magnitude with 2^bits-abs and reversing its sign.
            return (
                Self {
                    abs: self.abs.apply_negate_wrapping(bits),
                    is_positive: !self.is_positive,
                },
                true,
            );
        }

        let truncated_abs = self.abs.apply_wrapping(bits);
        if truncated_abs.is_zero() {
            return (Self::zero(), true);
        }
        let sign_bit = truncated_abs.get_bit(bits.wrapping_sub(1));

        let result = if self.is_positive {
            if sign_bit {
                // 0 < truncated_abs < 2^bits, so modular negation is exactly
                // 2^bits-truncated_abs and reuses the owned magnitude buffer.
                let abs = truncated_abs.apply_negate_wrapping(bits);
                Self {
                    abs,
                    is_positive: false,
                }
            } else {
                Self {
                    abs: truncated_abs,
                    is_positive: true,
                }
            }
        } else if !sign_bit || truncated_abs.is_power_of_two() {
            Self {
                abs: truncated_abs,
                is_positive: false,
            }
        } else {
            // The nonzero residue is below 2^bits; its modular complement is
            // the positive wrapped magnitude and fits the same allocation.
            let abs = truncated_abs.apply_negate_wrapping(bits);
            Self {
                abs,
                is_positive: true,
            }
        };
        (result, true)
    }
}
