//! Limb and 50-bit floating-point digit conversions for Harvey NTT.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer conversions into validated digit destination"
)]

use super::{LIMB_BITS, Limb, Ntt};

impl Ntt {
    /// Unpacks native limbs into 50-bit floating-point coefficients.
    ///
    /// # Safety
    /// `dst` has room for every emitted 50-bit digit.
    pub unsafe fn limbs_to_digits_50_into(dst: &mut [f64], limbs: &[Limb]) -> usize {
        let digit_mask = (1_u128 << 50).wrapping_sub(1);
        let dst_ptr = dst.as_mut_ptr();
        let mut count = 0_usize;
        let mut accumulator = 0_u128;
        let mut available_bits = 0_u32;

        for &limb in limbs {
            #[allow(
                clippy::as_conversions,
                reason = "Supported limb types fit in u64 and u128"
            )]
            let limb_u128 = limb as u128;
            accumulator |= limb_u128 << available_bits;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "LIMB_BITS is at most 64"
            )]
            let limb_bits = LIMB_BITS as u32;
            available_bits = available_bits.wrapping_add(limb_bits);

            while available_bits >= 50 {
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Masked to 50 bits"
                )]
                let word_raw = (accumulator & digit_mask) as u64;
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "50-bit integer is exact in f64"
                )]
                let word_float = word_raw as f64;
                // SAFETY: caller establishes dst capacity.
                unsafe {
                    *dst_ptr.add(count) = word_float;
                }
                count = count.wrapping_add(1);
                accumulator >>= 50;
                available_bits = available_bits.wrapping_sub(50);
            }
        }

        if available_bits > 0 && accumulator != 0 {
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Masked to 50 bits"
            )]
            let tail_word = (accumulator & digit_mask) as u64;
            #[allow(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "50-bit integer is exact in f64"
            )]
            let tail_float = tail_word as f64;
            // SAFETY: caller establishes dst capacity.
            unsafe {
                *dst_ptr.add(count) = tail_float;
            }
            count = count.wrapping_add(1);
        }

        // SAFETY: the caller's capacity contract proves count <= dst.len(),
        // so the padding range is fully in bounds. `fill` lowers to a
        // vectorized memset instead of a scalar store loop.
        unsafe {
            dst.get_unchecked_mut(count..).fill(0.0);
        }

        count
    }
}
