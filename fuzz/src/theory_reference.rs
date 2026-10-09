//! Independent finite-domain totient and Miller–Rabin references.

use std::sync::OnceLock;

use rug::{Integer, integer::IsPrime};

static SMALL_FACTOR_PRODUCT: OnceLock<Integer> = OnceLock::new();

pub struct TheoryReference;

impl TheoryReference {
    pub fn phi(value: u16) -> Option<Integer> {
        if value == 0 {
            return None;
        }
        let count = (1..=value)
            .filter(|&candidate| {
                let (mut a, mut b) = (value, candidate);
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                a == 1
            })
            .count();
        Some(Integer::from(count))
    }

    pub fn probably_prime(value: &Integer, rounds: u32) -> bool {
        if value < &2 {
            return false;
        }
        if value.to_u64().is_some() {
            return value.is_probably_prime(25) != IsPrime::No;
        }
        if value.is_even() {
            return false;
        }
        // gcd(n, 1999!) = 1 excludes every prime factor below 2000.
        let screen = SMALL_FACTOR_PRODUCT.get_or_init(|| Integer::from(Integer::factorial(1999)));
        if value.clone().gcd(screen) != 1 {
            return false;
        }
        let minus_one = value.clone() - 1_u32;
        let power = minus_one.find_one(0).unwrap();
        let odd = minus_one.clone() >> power;
        let mut prime = 2_u32;
        for _ in 0..rounds.clamp(1, 64) {
            while (2..prime).any(|divisor| prime.is_multiple_of(divisor)) {
                prime += 1;
            }
            let mut residue = Integer::from(prime).pow_mod(&odd, value).unwrap();
            if residue != 1 && residue != minus_one {
                let mut passed = false;
                for _ in 1..power {
                    residue = (residue.clone() * &residue) % value;
                    if residue == minus_one {
                        passed = true;
                        break;
                    }
                }
                if !passed {
                    return false;
                }
            }
            prime += 1;
        }
        true
    }
}
