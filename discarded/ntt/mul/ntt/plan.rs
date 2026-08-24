//! 50-bit floating-point NTT twiddle tables and planning.

#![allow(
    unsafe_code,
    clippy::suboptimal_flops,
    reason = "Twiddle table stores proved raw pointer writes and core in no_std does not provide f64::mul_add"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "Standard mathematical notation for FFT algorithms and Dekker TwoProduct"
)]

use core::{
    mem::{align_of, size_of},
    slice::from_raw_parts_mut,
};

use super::{LIMB_BITS, Limb};

const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;
const DEKKER_SPLIT: f64 = 134_217_729.0;

/// Zero-sized namespace for 50-bit floating-point Harvey NTT multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ntt;

impl Ntt {
    /// Computes the required scratch length in `f64` elements for 3-prime float50 multiplication ($9N$).
    #[must_use]
    #[inline]
    pub const fn scratch_len(transform_len: usize) -> usize {
        transform_len.wrapping_mul(9)
    }

    /// Computes the required scratch length in `f64` elements for 3-prime float50 squaring ($6N$).
    #[must_use]
    #[inline]
    pub const fn scratch_sqr_len(transform_len: usize) -> usize {
        transform_len.wrapping_mul(6)
    }

    /// Computes the required digit capacity for a given limb length and digit width.
    #[inline]
    #[must_use]
    pub fn digit_capacity(limb_len: usize, digit_bits: u32) -> Option<usize> {
        let digit_width = usize::try_from(digit_bits).ok()?;
        limb_len
            .checked_mul(LIMB_BITS)
            .map(|bits| bits.div_ceil(digit_width))
            .and_then(|digits| digits.checked_add(1))
    }

    /// Computes the minimum power-of-two transform length needed to convolve operands of given digit capacities.
    #[inline]
    #[must_use]
    pub fn transform_capacity(capacity_a: usize, capacity_b: usize) -> Option<usize> {
        let conv_len = capacity_a.checked_add(capacity_b)?.checked_sub(1)?;
        let len = conv_len.checked_next_power_of_two()?.max(16);
        Some(len)
    }

    /// Whether the 50-bit floating-point NTT tier admits the given operand lengths.
    #[inline]
    #[must_use]
    pub fn admits_mul(len_a: usize, len_b: usize) -> bool {
        if len_a == 0 || len_b == 0 {
            return true;
        }
        let Some(cap_a) = Self::digit_capacity(len_a, 50) else {
            return false;
        };
        let Some(cap_b) = Self::digit_capacity(len_b, 50) else {
            return false;
        };
        Self::transform_capacity(cap_a, cap_b).is_some()
    }

    /// Returns the scratch capacity in native `Limb`s required for NTT multiplication.
    #[must_use]
    pub fn mul_scratch_len(len_a: usize, len_b: usize) -> usize {
        let Some(cap_a) = Self::digit_capacity(len_a, 50) else {
            return 0;
        };
        let Some(cap_b) = Self::digit_capacity(len_b, 50) else {
            return 0;
        };
        let Some(transform_len) = Self::transform_capacity(cap_a, cap_b) else {
            return 0;
        };
        let f64_count = Self::scratch_len(transform_len);
        let f64_bytes = f64_count.saturating_mul(size_of::<f64>());
        let limb_bytes = size_of::<Limb>();
        f64_bytes.div_ceil(limb_bytes).saturating_add(1)
    }

    /// Returns the scratch capacity in native `Limb`s required for NTT squaring.
    #[must_use]
    pub fn sqr_scratch_len(len: usize) -> usize {
        let Some(cap) = Self::digit_capacity(len, 50) else {
            return 0;
        };
        let Some(transform_len) = Self::transform_capacity(cap, cap) else {
            return 0;
        };
        let f64_count = Self::scratch_sqr_len(transform_len);
        let f64_bytes = f64_count.saturating_mul(size_of::<f64>());
        let limb_bytes = size_of::<Limb>();
        f64_bytes.div_ceil(limb_bytes).saturating_add(1)
    }

    /// Whether the 50-bit floating-point NTT tier admits squaring the given operand length.
    #[inline]
    #[must_use]
    pub fn admits_sqr(len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let Some(cap) = Self::digit_capacity(len, 50) else {
            return false;
        };
        Self::transform_capacity(cap, cap).is_some()
    }

    /// Re-interprets a caller-owned `&mut [Limb]` buffer as an aligned `&mut [f64]` slice.
    #[allow(
        clippy::cast_ptr_alignment,
        reason = "Pointer is dynamically aligned to align_of::<f64>() via align_offset calculation"
    )]
    #[must_use]
    pub fn align_scratch_limbs_to_f64(
        limbs: &mut [Limb],
        required_f64s: usize,
    ) -> Option<&mut [f64]> {
        let ptr = limbs.as_mut_ptr();
        let align_offset = ptr.align_offset(align_of::<f64>());
        if align_offset > limbs.len() {
            return None;
        }
        let available_limbs = limbs.len().checked_sub(align_offset)?;
        let required_bytes = required_f64s.checked_mul(size_of::<f64>())?;
        let available_bytes = available_limbs.checked_mul(size_of::<Limb>())?;
        if available_bytes < required_bytes {
            return None;
        }
        // SAFETY:
        // - `aligned_ptr` is properly aligned to `align_of::<f64>()` as proved by `align_offset`.
        // - `required_f64s * size_of::<f64>()` bytes are contained within `limbs`.
        // - The borrow lifetime matches `&mut [Limb]`.
        unsafe {
            let aligned_ptr = ptr.add(align_offset).cast::<f64>();
            Some(from_raw_parts_mut(aligned_ptr, required_f64s))
        }
    }

    /// Exact scalar modular multiplication for 50-bit floating-point primes.
    #[inline]
    #[must_use]
    pub fn mulmod(a: f64, b: f64, prime: f64, pinv: f64) -> f64 {
        let (h_ab, l_ab) = two_product(a, b);
        let q = ((h_ab * pinv) + ROUND_MAGIC) - ROUND_MAGIC;
        let (h_qp, l_qp) = two_product(q, prime);
        let diff_h = h_ab - h_qp;
        (diff_h - l_qp) + l_ab
    }

    /// Generates forward DIF or inverse DIT twiddle factors for length `transform_len`.
    ///
    /// Precomputes twiddle triples `(W1, W2, W3)` for every radix-4 butterfly stage,
    /// eliminating runtime modular squaring in hot loops.
    #[allow(
        clippy::too_many_lines,
        reason = "Explicit radix-4 and radix-2 hybrid twiddle generation loop"
    )]
    pub fn generate_stage_twiddles(
        twiddles: &mut [f64],
        transform_len: usize,
        primitive_root: u64,
        prime_int: u64,
        prime: f64,
        pinv: f64,
        inverse: bool,
    ) {
        let p_minus_1 = prime_int.wrapping_sub(1);
        #[allow(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "primitive_root fits in f64"
        )]
        let root_f64 = primitive_root as f64;
        let base_root = if inverse {
            Self::pow_mod_float(root_f64, prime_int.wrapping_sub(2), prime, pinv)
        } else {
            root_f64
        };

        let mut offset = 0_usize;
        let tw_ptr = twiddles.as_mut_ptr();
        let is_odd_power_of_two = !transform_len.trailing_zeros().is_multiple_of(2);

        if inverse {
            // Inverse DIT: run radix-4 stages step = 4, 16, ..., then optional radix-2 final stage
            let max_r4_step = if is_odd_power_of_two {
                transform_len >> 1
            } else {
                transform_len
            };
            let mut step = 4_usize;
            while step <= max_r4_step {
                let quarter_len = step >> 2;
                let half_len = step >> 1;
                #[allow(
                    clippy::as_conversions,
                    reason = "step usize converts safely to u64 on supported pointer widths"
                )]
                let step_u64 = step as u64;
                let exp_step = p_minus_1.checked_div(step_u64).unwrap_or(1);
                let omega = Self::pow_mod_float(base_root, exp_step, prime, pinv);

                let mut current_w = 1.0_f64;
                for i in 0..half_len {
                    // SAFETY: offset + 3 * quarter_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        *tw_ptr.add(offset.wrapping_add(i)) = current_w;
                    }
                    current_w = Self::mulmod(current_w, omega, prime, pinv);
                }
                let q2 = quarter_len.wrapping_mul(2);
                for i in 0..quarter_len {
                    // SAFETY: offset + 3 * quarter_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        let w0 = *tw_ptr.add(offset.wrapping_add(i));
                        *tw_ptr.add(offset.wrapping_add(q2).wrapping_add(i)) =
                            Self::mulmod(w0, w0, prime, pinv);
                    }
                }
                let stage_len = quarter_len.wrapping_mul(3);
                offset = offset.wrapping_add(stage_len);
                step <<= 2;
            }
            if is_odd_power_of_two {
                let half_len = transform_len >> 1;
                #[allow(
                    clippy::as_conversions,
                    reason = "transform_len usize converts safely to u64 on supported pointer widths"
                )]
                let step_u64 = transform_len as u64;
                let exp_step = p_minus_1.checked_div(step_u64).unwrap_or(1);
                let omega = Self::pow_mod_float(base_root, exp_step, prime, pinv);
                let mut current_w = 1.0_f64;
                for i in 0..half_len {
                    // SAFETY: offset + half_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        *tw_ptr.add(offset.wrapping_add(i)) = current_w;
                    }
                    current_w = Self::mulmod(current_w, omega, prime, pinv);
                }
            }
        } else {
            // Forward DIF: optional radix-2 initial stage, then radix-4 stages
            if is_odd_power_of_two {
                let half_len = transform_len >> 1;
                #[allow(
                    clippy::as_conversions,
                    reason = "transform_len usize converts safely to u64 on supported pointer widths"
                )]
                let step_u64 = transform_len as u64;
                let exp_step = p_minus_1.checked_div(step_u64).unwrap_or(1);
                let omega = Self::pow_mod_float(base_root, exp_step, prime, pinv);
                let mut current_w = 1.0_f64;
                for i in 0..half_len {
                    // SAFETY: offset + half_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        *tw_ptr.add(offset.wrapping_add(i)) = current_w;
                    }
                    current_w = Self::mulmod(current_w, omega, prime, pinv);
                }
                offset = offset.wrapping_add(half_len);
            }
            let start_step = if is_odd_power_of_two {
                transform_len >> 1
            } else {
                transform_len
            };
            let mut step = start_step;
            while step >= 4 {
                let quarter_len = step >> 2;
                let half_len = step >> 1;
                #[allow(
                    clippy::as_conversions,
                    reason = "step usize converts safely to u64 on supported pointer widths"
                )]
                let step_u64 = step as u64;
                let exp_step = p_minus_1.checked_div(step_u64).unwrap_or(1);
                let omega = Self::pow_mod_float(base_root, exp_step, prime, pinv);

                let mut current_w = 1.0_f64;
                for i in 0..half_len {
                    // SAFETY: offset + 3 * quarter_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        *tw_ptr.add(offset.wrapping_add(i)) = current_w;
                    }
                    current_w = Self::mulmod(current_w, omega, prime, pinv);
                }
                let q2 = quarter_len.wrapping_mul(2);
                for i in 0..quarter_len {
                    // SAFETY: offset + 3 * quarter_len <= transform_len fits within twiddles buffer.
                    unsafe {
                        let w0 = *tw_ptr.add(offset.wrapping_add(i));
                        *tw_ptr.add(offset.wrapping_add(q2).wrapping_add(i)) =
                            Self::mulmod(w0, w0, prime, pinv);
                    }
                }
                let stage_len = quarter_len.wrapping_mul(3);
                offset = offset.wrapping_add(stage_len);
                step >>= 2;
            }
        }
    }

    /// Modular exponentiation for 50-bit floating point modulus.
    #[must_use]
    pub fn pow_mod_float(base: f64, exp: u64, prime: f64, pinv: f64) -> f64 {
        let mut res = 1.0_f64;
        let mut b = base;
        let mut e = exp;
        while e > 0 {
            if e & 1 != 0 {
                res = Self::mulmod(res, b, prime, pinv);
            }
            b = Self::mulmod(b, b, prime, pinv);
            e >>= 1;
        }
        res
    }
}

#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p_a = a * DEKKER_SPLIT;
    let a_hi = p_a - (p_a - a);
    let a_lo = a - a_hi;

    let p_b = b * DEKKER_SPLIT;
    let b_hi = p_b - (p_b - b);
    let b_lo = b - b_hi;

    let h = a * b;
    let l = ((a_hi * b_hi - h) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (h, l)
}
