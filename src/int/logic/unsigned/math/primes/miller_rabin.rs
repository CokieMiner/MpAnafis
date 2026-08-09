//! Trial division and Miller-Rabin kernels.

#![allow(
    unsafe_code,
    reason = "The primality kernels eliminate impossible zero-modulus branches after validating every modulus."
)]

use core::{cmp::Ordering, hint::unreachable_unchecked, mem::swap};

use super::{InternalMpUint, Limb, MontgomeryDomain, MulScratch, Primality};

const MR_U64_BASES: [u64; 7] = [2, 325, 9375, 28178, 450_775, 9_780_504, 1_795_265_022];

/// Simple trial division by the first few dozen primes.
impl Primality {
    pub fn trial_division(a: &InternalMpUint) -> bool {
        // Products of primes up to 241 (excluding 2)
        const PRIME_PRODUCTS: [u64; 6] = [
            16_294_579_238_595_022_365,
            7_145_393_598_349_078_859,
            6_408_001_374_760_705_163,
            690_862_709_424_854_779,
            4_312_024_209_383_942_993,
            57_599,
        ];

        let limbs = a.limbs();
        if limbs.is_empty() {
            return false;
        }

        let is_single = limbs.len() == 1;
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Limb safely casts to u64"
        )]
        let single_val = limbs.first().copied().unwrap_or(0) as u64;

        if is_single && single_val <= 241 {
            const SMALL_PRIMES: [u64; 52] = [
                3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
            ];
            return SMALL_PRIMES.contains(&single_val);
        }

        for &prod in &PRIME_PRODUCTS {
            let rem = if is_single {
                // SAFETY: prod is from PRIME_PRODUCTS, so it is > 0
                unsafe { single_val.checked_rem(prod).unwrap_unchecked() }
            } else {
                let mut r = 0_u128;
                let prod_128 = u128::from(prod);
                #[allow(clippy::as_conversions, reason = "Limb fits safely in u128")]
                for &limb in limbs.iter().rev() {
                    let shifted = r.wrapping_shl(Limb::BITS) | (limb as u128);
                    // SAFETY: prod is from PRIME_PRODUCTS (prime product), guaranteed non-zero.
                    r = unsafe { shifted.checked_rem(prod_128).unwrap_unchecked() };
                }
                #[allow(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "remainder always fits in u64"
                )]
                let rem_u64 = r as u64;
                rem_u64
            };

            if gcd_u64(rem, prod) > 1 {
                return false;
            }
        }

        if is_single && single_val < 241 * 241 {
            return true;
        }
        true
    }

    pub fn is_prime_u64(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n & 1 == 0 {
            return false;
        }

        let mut d = n.wrapping_sub(1);
        let mut s: u32 = 0;
        while d & 1 == 0 {
            d >>= 1;
            s = s.wrapping_add(1);
        }

        for &base in &MR_U64_BASES {
            let witness = base.checked_rem(n).unwrap_or_else(
                // SAFETY: n >= 2 is guaranteed above.
                || unsafe { unreachable_unchecked() },
            );
            if witness == 0 {
                continue;
            }
            if !miller_rabin_u64_round(witness, d, s, n) {
                return false;
            }
        }
        true
    }
}

fn miller_rabin_u64_round(base: u64, odd_part: u64, shift_count: u32, modulus: u64) -> bool {
    let mut residue = pow_mod_u64(base, odd_part, modulus);
    let modulus_minus_1 = modulus.wrapping_sub(1);
    if residue == 1 || residue == modulus_minus_1 {
        return true;
    }
    for _ in 1..shift_count {
        residue = mul_mod_u64(residue, residue, modulus);
        if residue == modulus_minus_1 {
            return true;
        }
    }
    false
}

fn pow_mod_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut acc = 1_u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod_u64(acc, base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod_u64(base, base, modulus);
        }
    }
    acc
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "u128 remainder modulo u64 always fits in u64"
)]
fn mul_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    let wide = u128::from(a).wrapping_mul(u128::from(b));
    let modulus_wide = u128::from(modulus);
    wide.checked_rem(modulus_wide).map_or_else(
        // SAFETY: all callers pass modulus n >= 2.
        || unsafe { unreachable_unchecked() },
        |rem| rem as u64,
    )
}

const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = match a.checked_rem(b) {
            Some(rem) => rem,
            // SAFETY: b != 0 is guaranteed by the while loop condition
            None => unsafe { unreachable_unchecked() },
        };
        a = t;
    }
    a
}

/// Single Miller-Rabin round: test base `a` on `n` with
/// `n-1 = d * 2^s`.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "standard math notation for MR test; scratch buffers passed for allocation reuse"
)]
impl Primality {
    #[must_use]
    pub fn miller_rabin_test(
        a: &InternalMpUint,
        d: &InternalMpUint,
        s: usize,
        n_minus_1_mont: &InternalMpUint,
        one_mont: &InternalMpUint,
        temp_prod: &mut InternalMpUint,
        temp_rem: &mut InternalMpUint,
        domain: &MontgomeryDomain,
        mul_scratch: &mut MulScratch,
    ) -> bool {
        let x_mont = domain.pow(a, d, mul_scratch, true);

        if x_mont.cmp(n_minus_1_mont) == Ordering::Equal {
            return true;
        }

        if x_mont.cmp(one_mont) == Ordering::Equal {
            return true;
        }

        if s <= 1 {
            return false;
        }

        let mut x_mont = x_mont;
        for _ in 0..s.wrapping_sub(1) {
            domain.mul_into_with_scratch(&x_mont, &x_mont, temp_rem, temp_prod, mul_scratch);
            swap(&mut x_mont, temp_rem);

            if x_mont.cmp(n_minus_1_mont) == Ordering::Equal {
                return true;
            }
        }
        false
    }
}
