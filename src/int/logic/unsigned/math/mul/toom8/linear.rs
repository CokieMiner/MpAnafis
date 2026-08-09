//! Fixed-width linear combinations and exact divisions for Toom-8 interpolation.

use core::cmp::min;

use super::{AddMulKernel, Addition, ArchKernels, LIMB_BITS, Limb, SharedEval, Toom8};

pub struct ScaledSource<'value> {
    pub value: &'value [Limb],
    pub scalar: i64,
}

impl Toom8 {
    pub fn linear_combination(
        dst: &mut [Limb],
        dst_scalar: i64,
        src: &[Limb],
        src_scalar: i64,
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        debug_assert_eq!(dst.len(), src.len(), "linear-combination widths differ");
        debug_assert_eq!(dst.len(), temporary.len(), "temporary width differs");
        let dst_magnitude = dst_scalar.unsigned_abs();
        let src_magnitude = src_scalar.unsigned_abs();
        if let (Ok(dst_word), Ok(src_word)) =
            (Limb::try_from(dst_magnitude), Limb::try_from(src_magnitude))
        {
            if dst_word != 1 {
                mul_word_modular_in_place(dst, dst_word);
            }
            if dst_scalar.is_negative() {
                negate_modular_in_place(dst);
            }
            if src_scalar.is_negative() {
                SharedEval::sub_mul_word_in_place(dst, src, src_word);
            } else {
                SharedEval::add_mul_word_with_kernel_in_place(dst, src, src_word, add_mul_kernel);
            }
            return;
        }

        // On a 16-bit Limb, some Toom-8 matrix constants span multiple limbs.
        // Factoring those constants preserves the same fixed-width modular value;
        // 32- and 64-bit targets take the single-word path above.
        multiply_u64_in_place(dst, dst_magnitude);
        if dst_scalar.is_negative() {
            negate_modular_in_place(dst);
        }
        temporary.copy_from_slice(src);
        multiply_u64_in_place(temporary, src_magnitude);
        if src_scalar.is_negative() {
            negate_modular_in_place(temporary);
        }
        let _ = Addition::add_slice_in_place(dst, temporary);
    }

    pub fn exact_sub_mul_u64_odd_in_place(
        dst: &mut [Limb],
        src: &[Limb],
        scalar: i64,
        divisor: u64,
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        assert!(scalar > 0, "fused subtraction scalar must be positive");
        assert!(divisor & 1 == 1, "fused exact divisor must be odd");
        if let (Ok(scalar_word), Ok(divisor_word)) =
            (Limb::try_from(scalar), Limb::try_from(divisor))
        {
            SharedEval::exact_sub_mul_word_odd_in_place(dst, src, scalar_word, divisor_word);
            return;
        }

        // Some Toom-8 constants exceed one limb on 16- and 32-bit targets. Keep
        // the portable factored path there; 64-bit targets use the fused pass.
        Self::linear_combination(
            dst,
            1,
            src,
            scalar.wrapping_neg(),
            temporary,
            add_mul_kernel,
        );
        Self::exact_signed_div_u64(dst, divisor);
    }

    pub fn exact_sub_mul_two_u64_odd_in_place(
        dst: &mut [Limb],
        sources: [ScaledSource<'_>; 2],
        divisor: u64,
        temporary: &mut [Limb],
        add_mul_kernel: AddMulKernel,
    ) {
        let [primary, secondary] = sources;
        let primary_src = primary.value;
        let primary_scalar = primary.scalar;
        let secondary_src = secondary.value;
        let secondary_scalar = secondary.scalar;
        assert!(primary_scalar > 0, "primary scalar must be positive");
        assert!(secondary_scalar > 0, "secondary scalar must be positive");
        assert!(divisor & 1 == 1, "exact divisor must be odd");
        if let (Ok(primary_word), Ok(secondary_word), Ok(divisor_word)) = (
            Limb::try_from(primary_scalar),
            Limb::try_from(secondary_scalar),
            Limb::try_from(divisor),
        ) {
            exact_sub_mul_two_word_odd_in_place(
                dst,
                primary_src,
                primary_word,
                secondary_src,
                secondary_word,
                divisor_word,
            );
            return;
        }

        Self::linear_combination(
            dst,
            1,
            primary_src,
            primary_scalar.wrapping_neg(),
            temporary,
            add_mul_kernel,
        );
        Self::exact_sub_mul_u64_odd_in_place(
            dst,
            secondary_src,
            secondary_scalar,
            divisor,
            temporary,
            add_mul_kernel,
        );
    }

    pub fn exact_signed_div_u64(value: &mut [Limb], divisor: u64) {
        assert!(divisor != 0, "exact Toom-8 divisor must be nonzero");
        if let Ok(word) = Limb::try_from(divisor) {
            exact_signed_div_limb(value, word);
            return;
        }
        let mut remaining = divisor;
        let mut factor = 2_u64;
        while factor <= remaining.div_euclid(factor) {
            while remaining.is_multiple_of(factor) {
                // SAFETY: factor <= divisor <= 2^64-1; div_euclid guarantees factor fits in Limb.
                let word = unsafe { Limb::try_from(factor).unwrap_unchecked() };
                exact_signed_div_limb(value, word);
                remaining = remaining.div_euclid(factor);
            }
            factor = factor.wrapping_add(1);
        }
        if remaining != 1 {
            // SAFETY: remaining < divisor <= 2^64-1; fits in Limb on 64-bit targets.
            let word = unsafe { Limb::try_from(remaining).unwrap_unchecked() };
            exact_signed_div_limb(value, word);
        }
    }

    pub fn exact_signed_div2_repeated(value: &mut [Limb], shifts: usize) {
        let mut remaining = shifts;
        let max_shift = LIMB_BITS.wrapping_sub(1);
        while remaining != 0 {
            let current = min(remaining, max_shift);
            // SAFETY: max_shift = LIMB_BITS - 1, so current < 2^32 always.
            let current_u32 = unsafe { u32::try_from(current).unwrap_unchecked() };
            SharedEval::exact_signed_div_power_of_two_in_place(value, current_u32);
            remaining = remaining.wrapping_sub(current);
        }
    }
}

fn multiply_u64_in_place(value: &mut [Limb], scalar: u64) {
    if let Ok(word) = Limb::try_from(scalar) {
        mul_word_modular_in_place(value, word);
        return;
    }
    let mut remaining = scalar;
    let mut factor = 2_u64;
    while factor <= remaining.div_euclid(factor) {
        while remaining.is_multiple_of(factor) {
            // SAFETY: factor is a factor of scalar <= 2^64; on every supported target Limb >= 16.
            let word = unsafe { Limb::try_from(factor).unwrap_unchecked() };
            mul_word_modular_in_place(value, word);
            remaining = remaining.div_euclid(factor);
        }
        factor = factor.wrapping_add(1);
    }
    if remaining != 1 {
        // SAFETY: remaining < scalar <= 2^64; on every supported target Limb >= 16.
        let word = unsafe { Limb::try_from(remaining).unwrap_unchecked() };
        mul_word_modular_in_place(value, word);
    }
}

fn exact_sub_mul_two_word_odd_in_place(
    dst: &mut [Limb],
    primary_src: &[Limb],
    primary_scalar: Limb,
    secondary_src: &[Limb],
    secondary_scalar: Limb,
    divisor: Limb,
) {
    debug_assert_eq!(dst.len(), primary_src.len(), "primary widths differ");
    debug_assert_eq!(dst.len(), secondary_src.len(), "secondary widths differ");
    debug_assert!(divisor & 1 == 1, "exact divisor must be odd");
    debug_assert!(
        primary_scalar.checked_add(secondary_scalar).is_some(),
        "combined scalar carry exceeds one limb"
    );
    let inverse = SharedEval::invert_odd(divisor);
    let mut product_carry = 0;
    let mut division_borrow = 0;
    for ((dst_limb, primary_limb), secondary_limb) in
        dst.iter_mut().zip(primary_src).zip(secondary_src)
    {
        let (primary_low, primary_high) =
            ArchKernels::mul_limb_lo_hi(*primary_limb, primary_scalar);
        let (secondary_low, secondary_high) =
            ArchKernels::mul_limb_lo_hi(*secondary_limb, secondary_scalar);
        let (product_sum, sum_overflow) = primary_low.overflowing_add(secondary_low);
        let (low_with_carry, carry_overflow) = product_sum.overflowing_add(product_carry);
        let (difference, subtraction_underflow) = dst_limb.overflowing_sub(low_with_carry);
        product_carry = primary_high
            .wrapping_add(secondary_high)
            .wrapping_add(Limb::from(sum_overflow))
            .wrapping_add(Limb::from(carry_overflow))
            .wrapping_add(Limb::from(subtraction_underflow));

        let (adjusted, division_underflow) = difference.overflowing_sub(division_borrow);
        let quotient = adjusted.wrapping_mul(inverse);
        let (_, quotient_high) = ArchKernels::mul_limb_lo_hi(quotient, divisor);
        division_borrow = quotient_high.wrapping_add(Limb::from(division_underflow));
        *dst_limb = quotient;
    }
    // The exact identity is evaluated modulo B^n; final carries are discarded
    // sign extension beyond the interpolation guard.
    let _ = (product_carry, division_borrow);
}

fn mul_word_modular_in_place(value: &mut [Limb], scalar: Limb) {
    if scalar == 0 {
        value.fill(0);
        return;
    }
    let mut carry = 0;
    for limb in value {
        let (product_low, product_high) = ArchKernels::mul_limb_lo_hi(*limb, scalar);
        let (with_carry, overflow) = product_low.overflowing_add(carry);
        *limb = with_carry;
        carry = product_high.wrapping_add(Limb::from(overflow));
    }
    // Signed interpolation is arithmetic modulo the fixed table width. For a
    // negative input the final carry is precisely sign extension and is dropped.
    let _ = carry;
}

fn exact_signed_div_limb(value: &mut [Limb], divisor: Limb) {
    let power_of_two = divisor.trailing_zeros();
    SharedEval::exact_signed_div_power_of_two_in_place(value, power_of_two);
    let odd_divisor = divisor.wrapping_shr(power_of_two);
    if odd_divisor == 255 {
        SharedEval::exact_div_radix_minus_one_in_place::<255>(value);
    } else if odd_divisor != 1 {
        SharedEval::exact_div_odd_in_place(value, odd_divisor, SharedEval::invert_odd(odd_divisor));
    }
}

fn negate_modular_in_place(value: &mut [Limb]) {
    let mut carry = 1;
    for limb in value {
        let (negated, overflow) = (!*limb).overflowing_add(carry);
        *limb = negated;
        carry = Limb::from(overflow);
    }
}
