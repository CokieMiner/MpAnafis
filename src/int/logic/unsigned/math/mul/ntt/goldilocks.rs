//! Single-prime NTT over the Goldilocks field `2^64 - 2^32 + 1`.

use alloc::{vec, vec::Vec};

use super::Ntt;

const PRIME: u64 = 0xffff_ffff_0000_0001;
const EPSILON: u64 = 0xffff_ffff;
const PRIMITIVE_ROOT: u64 = 7;

pub const PRIME_U128: u128 = 18_446_744_069_414_584_321;

impl Ntt {
    pub fn multiply_digits(
        a: &[u32],
        b: &[u32],
        convolution_len: usize,
        transform_len: usize,
        digit_bits: u32,
    ) -> Vec<u32> {
        let mut left = vec![0; transform_len];
        let mut right = vec![0; transform_len];
        for (dst, src) in left.iter_mut().zip(a) {
            *dst = u64::from(*src);
        }
        for (dst, src) in right.iter_mut().zip(b) {
            *dst = u64::from(*src);
        }
        forward_transform_pair(&mut left, &mut right);
        for (left_value, right_value) in left.iter_mut().zip(&right) {
            *left_value = mul_mod(*left_value, *right_value);
        }
        inverse_transform(&mut left);
        left.truncate(convolution_len);
        coefficients_to_digits(&left, digit_bits)
    }
}

fn forward_transform_pair(left: &mut [u64], right: &mut [u64]) {
    let mut block_len = left.len();
    while block_len >= 2 {
        // SAFETY: block_len is bounded by the transform size; usize ≤ u64 on all targets.
        let exponent = PRIME
            .wrapping_sub(1)
            .div_euclid(unsafe { u64::try_from(block_len).unwrap_unchecked() });
        let block_root = pow_mod(PRIMITIVE_ROOT, exponent);
        let half_len = block_len >> 1;
        for (left_block, right_block) in left
            .chunks_exact_mut(block_len)
            .zip(right.chunks_exact_mut(block_len))
        {
            let (left_low, left_high) = left_block.split_at_mut(half_len);
            let (right_low, right_high) = right_block.split_at_mut(half_len);
            let mut twiddle = 1;
            for ((left_low_value, left_high_value), (right_low_value, right_high_value)) in left_low
                .iter_mut()
                .zip(left_high)
                .zip(right_low.iter_mut().zip(right_high))
            {
                let left_lower = *left_low_value;
                let left_upper = *left_high_value;
                *left_low_value = add_mod(left_lower, left_upper);
                *left_high_value = mul_mod(sub_mod(left_lower, left_upper), twiddle);

                let right_lower = *right_low_value;
                let right_upper = *right_high_value;
                *right_low_value = add_mod(right_lower, right_upper);
                *right_high_value = mul_mod(sub_mod(right_lower, right_upper), twiddle);
                twiddle = mul_mod(twiddle, block_root);
            }
        }
        block_len >>= 1;
    }
}

fn inverse_transform(values: &mut [u64]) {
    let inverse_root = pow_mod(PRIMITIVE_ROOT, PRIME.wrapping_sub(2));
    let mut block_len = 2;
    while block_len <= values.len() {
        // SAFETY: block_len is bounded by the transform size; usize ≤ u64 on all targets.
        let exponent = PRIME
            .wrapping_sub(1)
            .div_euclid(unsafe { u64::try_from(block_len).unwrap_unchecked() });
        let block_root = pow_mod(inverse_root, exponent);
        let half_len = block_len >> 1;
        for block in values.chunks_exact_mut(block_len) {
            let (low, high) = block.split_at_mut(half_len);
            let mut twiddle = 1;
            for (low_value, high_value) in low.iter_mut().zip(high) {
                let upper = mul_mod(*high_value, twiddle);
                let lower = *low_value;
                *low_value = add_mod(lower, upper);
                *high_value = sub_mod(lower, upper);
                twiddle = mul_mod(twiddle, block_root);
            }
        }
        block_len = block_len.wrapping_mul(2);
    }
    // SAFETY: values.len() is the transform length, usize ≤ u64 on all targets.
    let inverse_len = pow_mod(
        unsafe { u64::try_from(values.len()).unwrap_unchecked() },
        PRIME.wrapping_sub(2),
    );
    for value in values {
        *value = mul_mod(*value, inverse_len);
    }
}

fn coefficients_to_digits(coefficients: &[u64], digit_bits: u32) -> Vec<u32> {
    let mask = (1_u128 << digit_bits).wrapping_sub(1);
    let mut result = Vec::with_capacity(coefficients.len().wrapping_add(8));
    let mut carry = 0_u128;
    for coefficient in coefficients {
        let with_carry = u128::from(*coefficient).wrapping_add(carry);
        // SAFETY: mask ensures the result fits in u32; digit_bits <= 31 for Goldilocks.
        result.push(unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() });
        carry = with_carry >> digit_bits;
    }
    while carry != 0 {
        // SAFETY: mask ensures the result fits in u32.
        result.push(unsafe { u32::try_from(carry & mask).unwrap_unchecked() });
        carry >>= digit_bits;
    }
    result
}

fn pow_mod(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, base);
        }
        base = mul_mod(base, base);
        exponent >>= 1;
    }
    result
}

#[allow(clippy::inline_always, reason = "Critical in hot NTT butterfly loop")]
#[inline(always)]
const fn add_mod(a: u64, b: u64) -> u64 {
    let (sum, overflowed) = a.overflowing_add(b);
    if overflowed {
        sum.wrapping_add(EPSILON)
    } else if sum >= PRIME {
        sum.wrapping_sub(PRIME)
    } else {
        sum
    }
}

#[allow(clippy::inline_always, reason = "Critical in hot NTT butterfly loop")]
#[inline(always)]
const fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a.wrapping_sub(b)
    } else {
        a.wrapping_sub(b).wrapping_sub(EPSILON)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The casts extract the exact low and high 64-bit halves of a u128 product"
)]
#[allow(
    clippy::inline_always,
    reason = "Critical for Goldilocks NTT butterfly reduction performance"
)]
#[inline(always)]
fn mul_mod(a: u64, b: u64) -> u64 {
    let product = u128::from(a).wrapping_mul(u128::from(b));
    let low = product as u64;
    let high = (product >> 64) as u64;
    let high_low = high & EPSILON;
    let high_high = high >> 32;
    // Because 2^64 == 2^32 - 1 modulo PRIME, the high word folds to
    // high_low*EPSILON - high_high. Since EPSILON == 2^32 - 1, we compute
    // high_low*EPSILON as (high_low << 32) - high_low to avoid a 64-bit multiply.
    let (low_minus_high, borrowed) = low.overflowing_sub(high_high);
    let folded_low = if borrowed {
        low_minus_high.wrapping_sub(EPSILON)
    } else {
        low_minus_high
    };
    let high_low_mul = (high_low << 32).wrapping_sub(high_low);
    add_mod(folded_low, high_low_mul)
}

#[cfg(test)]
#[path = "../tests/kernels/goldilocks.rs"]
mod tests;
