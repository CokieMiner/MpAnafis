//! Multi-prime Garner CRT reconstruction with division-free Barrett modular arithmetic.
//!
//! The reconstruction sweep intentionally remains scalar. Although each
//! residue pair is independent, every output digit consumes the previous
//! coefficient's carry. A useful AVX2 batch would therefore need a temporary
//! coefficient buffer and a separate serial carry pass. The two-prime path
//! also needs a 64-bit-by-32-bit Barrett quotient; AVX2's `_mm256_mul_epu32`
//! exposes only the low 32-bit factors and has no native high-half operation
//! for the required 64/128-bit product. The three-prime path widens to nearly
//! 2^94 during Barrett estimation and has the same limitation. Until a
//! measured scratch-buffer design supplies exact high halves, the scalar path
//! is the correctness and performance baseline; no truncating SIMD shortcut
//! is permitted.

use super::Ntt;

const FIRST_PRIME: u64 = 2_013_265_921;
const SECOND_PRIME: u64 = 1_811_939_329;
const THIRD_PRIME: u64 = 469_762_049;

const BARRETT_INV_SECOND: u128 = 2_545_165_803;
const BARRETT_INV_THIRD: u128 = 9_817_068_084;

const FIRST_INVERSE_SECOND: u64 = pow_mod(
    FIRST_PRIME % SECOND_PRIME,
    SECOND_PRIME.wrapping_sub(2),
    SECOND_PRIME,
);

const FIRST_SECOND: u64 = FIRST_PRIME.wrapping_mul(SECOND_PRIME);
const FIRST_SECOND_INVERSE_THIRD: u64 = pow_mod(
    FIRST_SECOND % THIRD_PRIME,
    THIRD_PRIME.wrapping_sub(2),
    THIRD_PRIME,
);

impl Ntt {
    pub fn reconstruct_two_slices(
        result: &mut [u32],
        first: &[u32],
        second: &[u32],
        digit_bits: u32,
    ) -> usize {
        let mask = digit_mask(digit_bits);
        let mut carry = 0_u128;
        let mut count = 0_usize;
        for ((first_value, second_value), dst) in first.iter().zip(second).zip(result.iter_mut()) {
            let first_value_u64 = u64::from(*first_value);
            let first_mod_second = if first_value_u64 >= SECOND_PRIME {
                first_value_u64.wrapping_sub(SECOND_PRIME)
            } else {
                first_value_u64
            };
            let second_val_u64 = u64::from(*second_value);
            let second_delta = if second_val_u64 >= first_mod_second {
                second_val_u64.wrapping_sub(first_mod_second)
            } else {
                second_val_u64
                    .wrapping_add(SECOND_PRIME)
                    .wrapping_sub(first_mod_second)
            };
            let second_factor = barrett_reduce_second(second_delta, FIRST_INVERSE_SECOND);
            // Garner reconstruction is in [0, p1*p2), and both primes are below
            // 2^31, so this exact representative is below 2^62.
            let coefficient = first_value_u64.wrapping_add(FIRST_PRIME.wrapping_mul(second_factor));
            let with_carry = u128::from(coefficient).wrapping_add(carry);
            // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
            *dst = unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() };
            carry = with_carry >> digit_bits;
            count = count.wrapping_add(1);
        }
        while carry != 0 && count < result.len() {
            if let Some(dst) = result.get_mut(count) {
                // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
                *dst = unsafe { u32::try_from(carry & mask).unwrap_unchecked() };
                carry >>= digit_bits;
                count = count.wrapping_add(1);
            }
        }
        count
    }

    pub fn reconstruct_three_slices(
        result: &mut [u32],
        first: &[u32],
        second: &[u32],
        third: &[u32],
        digit_bits: u32,
    ) -> usize {
        let mask = digit_mask(digit_bits);
        let mut carry = 0_u128;
        let mut count = 0_usize;
        for (((first_value, second_value), third_value), dst) in
            first.iter().zip(second).zip(third).zip(result.iter_mut())
        {
            let first_value_u64 = u64::from(*first_value);
            let first_mod_second = if first_value_u64 >= SECOND_PRIME {
                first_value_u64.wrapping_sub(SECOND_PRIME)
            } else {
                first_value_u64
            };
            let second_val_u64 = u64::from(*second_value);
            let second_delta = if second_val_u64 >= first_mod_second {
                second_val_u64.wrapping_sub(first_mod_second)
            } else {
                second_val_u64
                    .wrapping_add(SECOND_PRIME)
                    .wrapping_sub(first_mod_second)
            };
            let second_factor = barrett_reduce_second(second_delta, FIRST_INVERSE_SECOND);
            let first_two = first_value_u64.wrapping_add(FIRST_PRIME.wrapping_mul(second_factor));

            let first_two_mod_third = barrett_reduce_third(first_two, 1);

            let third_val_u64 = u64::from(*third_value);
            let third_delta = if third_val_u64 >= first_two_mod_third {
                third_val_u64.wrapping_sub(first_two_mod_third)
            } else {
                third_val_u64
                    .wrapping_add(THIRD_PRIME)
                    .wrapping_sub(first_two_mod_third)
            };
            let third_factor = barrett_reduce_third(third_delta, FIRST_SECOND_INVERSE_THIRD);
            let coefficient = u128::from(first_two)
                .wrapping_add(u128::from(FIRST_SECOND).wrapping_mul(u128::from(third_factor)));
            let with_carry = coefficient.wrapping_add(carry);
            // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
            *dst = unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() };
            carry = with_carry >> digit_bits;
            count = count.wrapping_add(1);
        }
        while carry != 0 && count < result.len() {
            if let Some(dst) = result.get_mut(count) {
                // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
                *dst = unsafe { u32::try_from(carry & mask).unwrap_unchecked() };
                carry >>= digit_bits;
                count = count.wrapping_add(1);
            }
        }
        count
    }
}

#[allow(
    clippy::inline_always,
    reason = "Tight Barrett reduction called once per digit during Garner CRT reconstruction"
)]
#[inline(always)]
fn barrett_reduce_second(a: u64, b: u64) -> u64 {
    let prod = a.wrapping_mul(b);
    let wide_q = (u128::from(prod).wrapping_mul(BARRETT_INV_SECOND)) >> 62;
    // SAFETY: prod < SECOND_PRIME^2 < 2^62 and BARRETT_INV_SECOND < 2^32,
    // so wide_q < 2^32 and therefore fits in u64.
    let q = unsafe { u64::try_from(wide_q).unwrap_unchecked() };
    let rem = prod.wrapping_sub(q.wrapping_mul(SECOND_PRIME));
    if rem >= SECOND_PRIME {
        rem.wrapping_sub(SECOND_PRIME)
    } else {
        rem
    }
}

#[allow(
    clippy::inline_always,
    reason = "Tight Barrett reduction called once per digit during Garner CRT reconstruction"
)]
#[inline(always)]
fn barrett_reduce_third(a: u64, b: u64) -> u64 {
    let prod = a.wrapping_mul(b);
    let wide_q = (u128::from(prod).wrapping_mul(BARRETT_INV_THIRD)) >> 62;
    // SAFETY: callers keep prod below 2^62 and BARRETT_INV_THIRD < 2^34,
    // so wide_q < 2^34 and therefore fits in u64.
    let q = unsafe { u64::try_from(wide_q).unwrap_unchecked() };
    let rem = prod.wrapping_sub(q.wrapping_mul(THIRD_PRIME));
    if rem >= THIRD_PRIME {
        rem.wrapping_sub(THIRD_PRIME)
    } else {
        rem
    }
}

#[allow(
    clippy::inline_always,
    reason = "Trivial const mask evaluation inlined into inner loop"
)]
#[inline(always)]
const fn digit_mask(digit_bits: u32) -> u128 {
    (1_u128 << digit_bits).wrapping_sub(1)
}

const fn pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

const fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    // CRT operands are below 2^31, so their product is exact below 2^62.
    a.wrapping_mul(b).rem_euclid(modulus)
}
