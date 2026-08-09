//! Signed two's-complement wrapping implemented on `InternalMpInt`.

use super::{InternalMpInt, InternalMpUint};

impl InternalMpInt {
    /// Wraps this value to `bits` two's-complement bits.
    #[inline]
    #[must_use]
    pub fn apply_wrapping(self, bits: usize) -> Self {
        let significant_bits = self.abs.significant_bits();
        if self.is_positive && significant_bits <= bits.saturating_sub(1) {
            return self;
        }
        if !self.is_positive {
            if self.abs.is_power_of_two() && significant_bits == bits {
                return self;
            }
            if significant_bits < bits {
                return self;
            }
        }

        let truncated_abs = self.abs.apply_wrapping(bits);
        if truncated_abs.is_zero() {
            return Self::zero();
        }
        let sign_bit = truncated_abs.get_bit(bits.wrapping_sub(1));

        if self.is_positive {
            if sign_bit {
                let two_pow_n = InternalMpUint::power_of_two(bits);
                // Width truncation proves `truncated_abs < 2^bits`.
                let abs = two_pow_n.sub(&truncated_abs);
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
            let two_pow_n = InternalMpUint::power_of_two(bits);
            // Width truncation proves `truncated_abs < 2^bits`.
            let abs = two_pow_n.sub(&truncated_abs);
            Self {
                abs,
                is_positive: true,
            }
        }
    }
}
