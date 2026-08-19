//! Montgomery field arithmetic and limb/digit conversion for NTT products.

use crate::parallel::ParallelExecutor;

use super::{ArchKernels, Limb, Modulus, Ntt};

impl Ntt {
    /// Unpack limbs into digit coefficients of width `digit_bits`.
    ///
    /// # Safety
    /// `digit_bits` is in `1..=31`, and `dst` has room for every emitted digit
    /// (including a possible final partial digit).
    pub unsafe fn limbs_to_digits_into(dst: &mut [u32], limbs: &[Limb], digit_bits: u32) -> usize {
        debug_assert!(
            (1..=31).contains(&digit_bits),
            "digit conversion requires a 1..=31 bit width"
        );
        // Four 16-bit coefficients fit exactly in each 64-bit limb.  The
        // architecture backend is selected once here, after the destination
        // capacity proves that its write-only contract is satisfied.  Other
        // widths retain the portable bit accumulator because their digits
        // cross native-limb boundaries in different ways.
        if digit_bits == 16
            && Limb::BITS == 64
            && limbs
                .len()
                .checked_mul(4)
                .is_some_and(|needed| dst.len() >= needed)
        {
            // SAFETY: the width and capacity guards above prove the backend's
            // 64-bit input and complete output-span preconditions.
            return unsafe {
                ArchKernels::ntt_digits_16_into(
                    dst.as_mut_ptr(),
                    limbs.as_ptr(),
                    limbs.len(),
                    dst.len(),
                )
            };
        }

        let digit_mask = (1_u128 << digit_bits).wrapping_sub(1);
        let dst_ptr = dst.as_mut_ptr();
        let mut count = 0_usize;
        let mut accumulator = 0_u128;
        let mut available_bits = 0_u32;
        let digit_bits_u128 = u128::from(digit_bits);

        for limb in limbs {
            // SAFETY: every supported native limb is at most 64 bits, so it
            // always converts to u128.
            accumulator |= unsafe { u128::try_from(*limb).unwrap_unchecked() } << available_bits;
            available_bits = available_bits.wrapping_add(Limb::BITS);
            while available_bits >= digit_bits {
                // SAFETY: the caller proves capacity for every emitted
                // digit; the mask proves the conversion fits in u32.
                unsafe {
                    *dst_ptr.add(count) =
                        u32::try_from(accumulator & digit_mask).unwrap_unchecked();
                }
                count = count.wrapping_add(1);
                accumulator >>= digit_bits_u128;
                available_bits = available_bits.wrapping_sub(digit_bits);
            }
        }
        if available_bits != 0 {
            // SAFETY: capacity is caller-proven; available_bits < digit_bits
            // ≤ 31, so the final accumulator fits in u32.
            unsafe {
                *dst_ptr.add(count) = u32::try_from(accumulator).unwrap_unchecked();
            }
            count = count.wrapping_add(1);
        }
        while count > 0 {
            // SAFETY: count is bounded by the proven output capacity.
            if unsafe { *dst_ptr.add(count.wrapping_sub(1)) != 0 } {
                break;
            }
            count = count.wrapping_sub(1);
        }
        count
    }

    /// Pack digit coefficients into limbs.
    ///
    /// # Safety
    /// `digit_bits` is in `1..=31`, and `dst` can hold all nonzero bits
    /// represented by `digits`. Zero padding in a final partial digit may
    /// extend beyond the destination's normalized width.
    pub unsafe fn digits_to_limbs(dst: &mut [Limb], digits: &[u32], digit_bits: u32) {
        debug_assert!(
            (1..=31).contains(&digit_bits),
            "digit conversion requires a 1..=31 bit width"
        );
        // This check is intentionally outside the coefficient loop.  A bad
        // transform/CRT count must fail at the conversion boundary before an
        // unchecked store can escape the validated destination, rather than
        // silently truncating or corrupting the allocator metadata.
        #[cfg(debug_assertions)]
        {
            let limb_bits = usize::try_from(Limb::BITS).unwrap_or(usize::MAX);
            let digit_width = usize::try_from(digit_bits).unwrap_or(usize::MAX);
            let required_limbs = digits
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, &digit)| {
                    (digit != 0).then(|| {
                        let used_bits = usize::try_from(digit.bit_width()).unwrap_or(usize::MAX);
                        index
                            .checked_mul(digit_width)
                            .and_then(|bits| bits.checked_add(used_bits))
                            .map_or(usize::MAX, |bits| bits.div_ceil(limb_bits))
                    })
                })
                .unwrap_or(0);
            debug_assert!(
                required_limbs <= dst.len(),
                "digit stream exceeds the validated destination width"
            );
        }
        dst.fill(0);
        let dst_ptr = dst.as_mut_ptr();
        let limb_mask = (1_u128 << Limb::BITS).wrapping_sub(1);
        let mut limb_count = 0_usize;
        let mut accumulator = 0_u128;
        let mut available_bits = 0_u32;
        let limb_bits_u128 = u128::from(Limb::BITS);

        // When digit_bits <= Limb::BITS (always true on 32-bit and 64-bit targets),
        // available_bits can only exceed Limb::BITS by at most digit_bits, so at
        // most one limb is drained per digit iteration.
        if digit_bits <= Limb::BITS {
            for digit in digits {
                accumulator |= u128::from(*digit) << available_bits;
                available_bits = available_bits.wrapping_add(digit_bits);
                if available_bits >= Limb::BITS {
                    // SAFETY: the caller proves destination capacity; the mask
                    // proves the conversion fits in one native limb.
                    unsafe {
                        *dst_ptr.add(limb_count) =
                            Limb::try_from(accumulator & limb_mask).unwrap_unchecked();
                    }
                    limb_count = limb_count.wrapping_add(1);
                    accumulator >>= limb_bits_u128;
                    available_bits = available_bits.wrapping_sub(Limb::BITS);
                }
            }
        } else {
            for digit in digits {
                accumulator |= u128::from(*digit) << available_bits;
                available_bits = available_bits.wrapping_add(digit_bits);
                while available_bits >= Limb::BITS {
                    // SAFETY: the caller proves destination capacity; the mask
                    // proves the conversion fits in one native limb.
                    unsafe {
                        *dst_ptr.add(limb_count) =
                            Limb::try_from(accumulator & limb_mask).unwrap_unchecked();
                    }
                    limb_count = limb_count.wrapping_add(1);
                    accumulator >>= limb_bits_u128;
                    available_bits = available_bits.wrapping_sub(Limb::BITS);
                }
            }
        }
        if available_bits != 0 && accumulator != 0 {
            debug_assert!(
                limb_count < dst.len(),
                "digit conversion destination is too narrow for a nonzero high limb"
            );
            // SAFETY: the validated destination contract and debug assertion
            // prove a final limb slot; available_bits is below Limb::BITS.
            unsafe {
                *dst_ptr.add(limb_count) = Limb::try_from(accumulator).unwrap_unchecked();
            }
        }
    }

    pub fn montgomery_pow(mut base: u32, mut exponent: u32, modulus: Modulus) -> u32 {
        let mut result = Self::to_montgomery(1, modulus);
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = Self::montgomery_mul(result, base, modulus);
            }
            base = Self::montgomery_mul(base, base, modulus);
            exponent >>= 1;
        }
        result
    }

    pub fn to_montgomery(value: u32, modulus: Modulus) -> u32 {
        Self::montgomery_mul(value, modulus.radix_squared, modulus)
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Montgomery reduction intentionally extracts the low radix word; REDC then proves the quotient is below 2p < 2^32"
    )]
    pub fn montgomery_mul(a: u32, b: u32, modulus: Modulus) -> u32 {
        let product = u64::from(a).wrapping_mul(u64::from(b));
        let factor = (product as u32).wrapping_mul(modulus.neg_inverse);
        let reduced = product
            .wrapping_add(u64::from(factor).wrapping_mul(u64::from(modulus.prime)))
            .wrapping_shr(32);
        let reduced_u32 = reduced as u32;
        if reduced_u32 >= modulus.prime {
            reduced_u32.wrapping_sub(modulus.prime)
        } else {
            reduced_u32
        }
    }

    /// Single-prime NTT convolution using the supplied execution policy.
    pub fn convolve_mod_slice_with_executor<E: ParallelExecutor>(
        left: &mut [u32],
        right: &mut [u32],
        a: &[u32],
        b: &[u32],
        modulus: Modulus,
        twiddle_buf: &mut [u32],
        executor: &E,
    ) {
        left.fill(0);
        right.fill(0);
        for (dst, src) in left.iter_mut().zip(a) {
            *dst = src.rem_euclid(modulus.prime);
        }
        for (dst, src) in right.iter_mut().zip(b) {
            *dst = src.rem_euclid(modulus.prime);
        }
        Self::forward_transform_pair_with_executor(left, right, modulus, twiddle_buf, executor);
        Self::pointwise_monty_mul_with_executor(left, right, modulus, executor);
        Self::inverse_transform_with_executor(left, modulus, twiddle_buf, executor);
    }

    /// Single-prime NTT square using the supplied execution policy.
    pub fn square_mod_slice_with_executor<E: ParallelExecutor>(
        values: &mut [u32],
        a: &[u32],
        modulus: Modulus,
        twiddle_buf: &mut [u32],
        executor: &E,
    ) {
        values.fill(0);
        for (dst, src) in values.iter_mut().zip(a) {
            *dst = src.rem_euclid(modulus.prime);
        }
        Self::forward_transform_single_with_executor(values, modulus, twiddle_buf, executor);
        Self::pointwise_monty_sqr_with_executor(values, modulus, executor);
        Self::inverse_transform_with_executor(values, modulus, twiddle_buf, executor);
    }
}
