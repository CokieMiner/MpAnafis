//! Core signed integer representation and two's-complement conversion helpers.

#![expect(
    unsafe_code,
    reason = "Bounded two's-complement prefixes and sign-bit normalization prove initialized access and exact truncation"
)]

use super::{InternalMpUint, LIMB_BITS, Limb};

/// Core signed multi-precision integer representation.
///
/// The magnitude is normalized. Zero has a positive sign; negative zero is invalid.
#[derive(Debug, Clone)]
pub struct InternalMpInt {
    /// Unsigned magnitude $|x|$ of the integer.
    pub abs: InternalMpUint,
    /// Sign flag: `true` if $x \ge 0$, `false` if $x < 0$.
    ///
    /// Must be `true` whenever `abs.is_zero()`.
    pub is_positive: bool,
}

impl InternalMpInt {
    /// Constructs canonical positive zero.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            abs: InternalMpUint::zero(),
            is_positive: true,
        }
    }

    /// Constructs positive one.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self {
            abs: InternalMpUint::one(),
            is_positive: true,
        }
    }

    /// Constructs the signed maximum $2^{width-1}-1$.
    ///
    /// Precondition: `width > 0`, validated by the public precision boundary.
    #[inline]
    #[must_use]
    pub fn max_for_bits(width: usize) -> Self {
        debug_assert!(width > 0, "signed maximum requires a nonzero width");
        Self {
            abs: InternalMpUint::max_for_bits(width.saturating_sub(1)),
            is_positive: true,
        }
    }

    /// Constructs the signed minimum $-2^{width-1}$.
    ///
    /// Precondition: `width > 0`, validated by the public precision boundary.
    #[inline]
    #[must_use]
    pub fn min_for_bits(width: usize) -> Self {
        debug_assert!(width > 0, "signed minimum requires a nonzero width");
        Self {
            // The single set bit constructs a normalized nonzero magnitude.
            abs: InternalMpUint::power_of_two(width.saturating_sub(1)),
            is_positive: false,
        }
    }

    /// Reserves magnitude storage for `limbs` words and initializes positive zero.
    #[inline]
    #[must_use]
    pub fn with_capacity(limbs: usize) -> Self {
        Self {
            abs: InternalMpUint::with_capacity(limbs),
            is_positive: true,
        }
    }

    /// Returns the minimum two's-complement bit width required to store this signed integer.
    ///
    /// For integer $x \in \mathbb{Z}$ with significant magnitude bits $k = \operatorname{sig}(|x|)$:
    /// - If $x \ge 0$, requires $k + 1$ bits to ensure a non-negative sign bit $0$.
    /// - If $x < 0$ and $|x| = 2^{k-1}$ is an exact power of two, fits exactly in $k$ bits (the minimum value $-2^{k-1}$).
    /// - If $x < 0$ and $|x| \ne 2^{k-1}$, requires $k + 1$ bits.
    ///
    /// Saturates at `usize::MAX`, which no bounded precision admits. Addressable
    /// magnitude storage can require an unrepresentable signed width on narrow
    /// targets; saturation preserves rejection by every bounded-width caller.
    #[must_use]
    pub fn required_signed_bits_for_bounded_storage(&self) -> usize {
        let sig = self.abs.significant_bits();
        if self.is_positive {
            sig.saturating_add(1)
        } else if self.abs.is_power_of_two() {
            sig
        } else {
            sig.saturating_add(1)
        }
    }

    /// Returns whether shifting this value left exceeds a caller-proved
    /// bounded signed width.
    #[inline]
    #[must_use]
    pub fn bounded_shl_overflows(&self, bits: usize, shift: usize) -> bool {
        let value_bits = self.required_signed_bits_for_bounded_storage();
        debug_assert!(
            value_bits <= bits,
            "bounded signed value must fit its declared precision"
        );
        // For non-zero x, signed storage requires `value_bits + shift` bits.
        // The caller proves `value_bits <= bits`, so the wrapping subtraction
        // equals the exact non-negative slack and avoids a second checked path.
        !self.abs.is_zero() && shift > bits.wrapping_sub(value_bits)
    }

    /// Returns whether a value fitting the signed `width_bits`-bit interval is
    /// its minimum endpoint $-2^{width\_bits-1}$.
    ///
    /// The caller must establish that this value fits the nonzero signed width.
    #[must_use]
    pub fn is_signed_min_for_width(&self, width_bits: usize) -> bool {
        debug_assert!(
            self.required_signed_bits_for_bounded_storage() <= width_bits,
            "signed endpoint input must fit its declared width"
        );
        if self.is_positive {
            return false;
        }
        // A fitting negative magnitude is at most 2^(width_bits-1). Reaching
        // width_bits significant bits therefore proves equality with that
        // endpoint; testing its power-of-two shape would repeat the fit proof.
        self.abs.significant_bits() == width_bits
    }

    /// Encodes this signed integer into a $w$-bit two's-complement unsigned magnitude.
    ///
    /// Precondition: `width > 0`.
    #[must_use]
    pub fn to_tc_bits(&self, width: usize) -> InternalMpUint {
        debug_assert!(width > 0, "two's-complement width must be non-zero");
        let limbs = self.abs.limbs();
        if width <= LIMB_BITS {
            // Reduction to one limb depends only on the low source limb.
            // Its native additive inverse equals the low limb of -abs.
            let low = limbs.first().copied().unwrap_or(0);
            let residue = if self.is_positive {
                low
            } else {
                low.wrapping_neg()
            };
            // 0 < width <= LIMB_BITS bounds this bit index by LIMB_BITS - 1,
            // so subtraction is exact in the native ring on every pointer width.
            let shift = LIMB_BITS.wrapping_sub(width);
            return InternalMpUint::from_limb(residue & (Limb::MAX >> shift));
        }
        let keep = width.div_ceil(LIMB_BITS).min(limbs.len());
        let magnitude = if keep < limbs.len() {
            // Reduction modulo 2^width depends only on the retained prefix.
            // SAFETY: keep is the minimum of the requested limb count and
            // limbs.len(); this branch retains a strict initialized prefix.
            InternalMpUint::from_limbs_slice(unsafe { limbs.get_unchecked(..keep) })
        } else {
            self.abs.clone()
        };
        if self.is_positive {
            magnitude.apply_wrapping(width)
        } else {
            magnitude.apply_negate_wrapping(width)
        }
    }

    /// Reconstructs a signed integer from $w$-bit two's-complement bits.
    ///
    /// The caller must establish `width > 0`. Zero is decoded with a positive sign.
    #[must_use]
    pub fn from_tc_bits(bits: InternalMpUint, width: usize) -> Self {
        debug_assert!(width > 0, "two's-complement width must be non-zero");
        let wrapped_bits = bits.apply_wrapping(width);
        if wrapped_bits.is_zero() {
            return Self::zero();
        }
        // Bit width - 1 is the two's-complement sign bit.
        if wrapped_bits.get_bit(width.wrapping_sub(1)) {
            let mut abs = wrapped_bits;
            let limbs = abs.limbs_mut();
            let full_len = limbs.len();
            // The set sign bit proves full_len > 0, so this predecessor index
            // is exact in the native ring and remains within the limb span.
            let top_index = full_len.wrapping_sub(1);
            // SAFETY: top_index < full_len bounds the initialized read.
            let mut top = !unsafe { *limbs.get_unchecked(top_index) };
            let partial = width.rem_euclid(LIMB_BITS);
            if partial != 0 {
                // 0 < partial < LIMB_BITS bounds the exact native-ring mask index.
                top &= Limb::MAX >> LIMB_BITS.wrapping_sub(partial);
            }
            let mut result_len = full_len;
            let complement_len = if top == 0 {
                // High all-ones source limbs complement to zero. Omit their
                // writes and retain only the normalized complement prefix.
                result_len = top_index;
                while result_len > 0 {
                    // 0 < result_len <= top_index bounds this exact predecessor index.
                    let index = result_len.wrapping_sub(1);
                    // SAFETY: index < full_len; the source limb is initialized.
                    if unsafe { *limbs.get_unchecked(index) } != Limb::MAX {
                        break;
                    }
                    result_len = index;
                }
                result_len
            } else {
                // SAFETY: top_index < full_len; this exclusive write retains
                // the nonzero, width-masked top complement limb.
                unsafe {
                    *limbs.get_unchecked_mut(top_index) = top;
                }
                top_index
            };
            // SAFETY: complement_len <= full_len bounds the initialized
            // mutable prefix, disjoint from the already processed top limb.
            for limb in unsafe { limbs.get_unchecked_mut(..complement_len) } {
                *limb = !*limb;
            }
            // SAFETY: result_len <= full_len only truncates initialized
            // storage. Its top is proved nonzero, or result_len is zero.
            unsafe {
                abs.set_len(result_len);
            }
            // -r mod 2^width = (~r mod 2^width)+1. The set sign bit bounds
            // this magnitude by 2^(width-1), within the original allocation.
            abs.increment();
            Self {
                abs,
                is_positive: false,
            }
        } else {
            Self {
                abs: wrapped_bits,
                is_positive: true,
            }
        }
    }
}
