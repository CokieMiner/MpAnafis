//! Exact two- and three-prime CRT reconstruction for NTT coefficients.

use alloc::vec::Vec;

use super::{MODULI, Ntt};

impl Ntt {
    pub fn reconstruct_two_into(
        result: &mut Vec<u32>,
        first: &[u32],
        second: &[u32],
        digit_bits: u32,
    ) {
        result.clear();
        let first_prime = u64::from(MODULI[0].prime);
        let second_prime = u64::from(MODULI[1].prime);
        let first_inverse_second = pow_mod(
            first_prime.rem_euclid(second_prime),
            second_prime.wrapping_sub(2),
            second_prime,
        );
        let mask = digit_mask(digit_bits);
        let mut carry = 0_u128;
        for (first_value, second_value) in first.iter().zip(second) {
            let first_value_u64 = u64::from(*first_value);
            let first_mod_second = first_value_u64.rem_euclid(second_prime);
            let second_delta = sub_mod(u64::from(*second_value), first_mod_second, second_prime);
            let second_factor = mul_mod(second_delta, first_inverse_second, second_prime);
            // Garner reconstruction is in [0, p1*p2), and both primes are below
            // 2^31, so this exact representative is below 2^62.
            let coefficient = first_value_u64.wrapping_add(first_prime.wrapping_mul(second_factor));
            let with_carry = u128::from(coefficient).wrapping_add(carry);
            // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
            result.push(unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() });
            carry = with_carry >> digit_bits;
        }
        append_carry(result, carry, mask, digit_bits);
    }

    pub fn reconstruct_three_into(
        result: &mut Vec<u32>,
        first: &[u32],
        second: &[u32],
        third: &[u32],
        digit_bits: u32,
    ) {
        result.clear();
        let first_prime = u64::from(MODULI[0].prime);
        let second_prime = u64::from(MODULI[1].prime);
        let third_prime = u64::from(MODULI[2].prime);
        let first_inverse_second = pow_mod(
            first_prime.rem_euclid(second_prime),
            second_prime.wrapping_sub(2),
            second_prime,
        );
        let first_second = first_prime.wrapping_mul(second_prime);
        let first_second_inverse_third = pow_mod(
            first_second.rem_euclid(third_prime),
            third_prime.wrapping_sub(2),
            third_prime,
        );
        let mask = digit_mask(digit_bits);
        let mut carry = 0_u128;
        for ((first_value, second_value), third_value) in first.iter().zip(second).zip(third) {
            let first_value_u64 = u64::from(*first_value);
            let first_mod_second = first_value_u64.rem_euclid(second_prime);
            let second_delta = sub_mod(u64::from(*second_value), first_mod_second, second_prime);
            let second_factor = mul_mod(second_delta, first_inverse_second, second_prime);
            let first_two = first_value_u64.wrapping_add(first_prime.wrapping_mul(second_factor));
            let third_delta = sub_mod(
                u64::from(*third_value),
                first_two.rem_euclid(third_prime),
                third_prime,
            );
            let third_factor = mul_mod(third_delta, first_second_inverse_third, third_prime);
            let coefficient = u128::from(first_two)
                .wrapping_add(u128::from(first_second).wrapping_mul(u128::from(third_factor)));
            let with_carry = coefficient.wrapping_add(carry);
            // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
            result.push(unsafe { u32::try_from(with_carry & mask).unwrap_unchecked() });
            carry = with_carry >> digit_bits;
        }
        append_carry(result, carry, mask, digit_bits);
    }
}

fn append_carry(result: &mut Vec<u32>, mut carry: u128, mask: u128, digit_bits: u32) {
    while carry != 0 {
        // SAFETY: mask limits to digit_bits ≤ 31, always fits in u32.
        result.push(unsafe { u32::try_from(carry & mask).unwrap_unchecked() });
        carry >>= digit_bits;
    }
}

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

const fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    if a >= b {
        a.wrapping_sub(b)
    } else {
        modulus.wrapping_sub(b.wrapping_sub(a))
    }
}
