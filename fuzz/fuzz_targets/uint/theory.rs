//! Number theory, powers, and modular arithmetic checks for `ArbiUint`.

use arbi_anafis::ArbiUint;
use rug::{Integer, ops::Pow};

pub fn fuzz_all(arbi_a: &ArbiUint, arbi_b: &ArbiUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 14 {
        0 => {
            if let Some((g, l)) = arbi_a.gcd_lcm(arbi_b) {
                let rug_g = rug_a.clone().gcd(rug_b);
                let rug_l = rug_a.clone().lcm(rug_b);
                assert_eq!(format!("{:x}", g), format!("{:x}", rug_g));
                assert_eq!(format!("{:x}", l), format!("{:x}", rug_l));
                assert_eq!(arbi_a.gcd(arbi_b), g);
                if let Some(l_only) = arbi_a.lcm(arbi_b) {
                    assert_eq!(l_only, l);
                }
                assert_eq!(arbi_a.is_coprime(arbi_b), g == ArbiUint::one());
            }
        }
        1 => {
            if *rug_b != 0 && let Some((g, _x, _y)) = arbi_a.extended_gcd(arbi_b) {
                assert_eq!(g, arbi_a.gcd(arbi_b));
            }
        }
        2 => {
            if let Some(root) = arbi_a.isqrt() {
                let root_sq = root.clone() * root.clone();
                let next_sq = (root.clone() + ArbiUint::one()) * (root.clone() + ArbiUint::one());
                assert!(root_sq <= *arbi_a);
                assert!(*arbi_a <= next_sq || *arbi_a == ArbiUint::zero());
                assert_eq!(arbi_a.is_perfect_square(), root_sq == *arbi_a);
            }
        }
        3 => {
            let n = (arbi_b.to_u64().unwrap_or(0) % 10) as u32;
            if n > 0 && let Some(root) = arbi_a.nth_root(n) {
                let root_pow = root.clone().pow(n);
                assert!(root_pow <= *arbi_a);
            }
        }
        4 => {
            if let Some((s, r)) = arbi_a.sqrt_rem() {
                let sq = s.clone() * s.clone() + r;
                assert_eq!(sq, *arbi_a);
            }
        }
        5 => {
            let exp = (arbi_b.to_u64().unwrap_or(0) % 16) as u32;
            let arbi = arbi_a.pow(exp);
            let rug = Integer::from(rug_a.pow(exp));
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            assert_eq!(arbi_a.checked_pow(exp), Some(arbi));
        }
        6 => {
            if *rug_b > 1 && let Some(arbi) = arbi_a.add_mod(arbi_b, arbi_b) {
                let rug = Integer::from(rug_a + rug_b) % rug_b;
                assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            }
        }
        7 => {
            if *rug_b > 1 && let Some(arbi) = arbi_a.mul_mod(arbi_b, arbi_b) {
                let rug = Integer::from(rug_a * rug_b) % rug_b;
                assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            }
        }
        8 => {
            if *rug_b > 1 {
                let exp = arbi_a.clone() % ArbiUint::from(256_u32);
                let rug_exp = rug_a.clone() % 256;
                if let Some(arbi) = arbi_a.pow_mod(&exp, arbi_b) {
                    let rug = match rug_a.clone().pow_mod(&rug_exp, rug_b) {
                        Ok(val) => val,
                        Err(_) => unreachable!(),
                    };
                    assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
                }
            }
        }
        9 => {
            if *rug_b > 1
                && let Some(arbi) = arbi_a.invert(arbi_b)
                && let Ok(rug) = rug_a.clone().invert(rug_b)
            {
                assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            }
        }
        10 => {
            if *rug_b > 1 && let Some(arbi) = arbi_a.sub_mod(arbi_a, arbi_b) {
                let mut rug = Integer::from(0);
                if rug < 0 {
                    rug += rug_b;
                }
                rug %= rug_b;
                assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            }
        }
        11 => {
            let arbi_prime = arbi_a.is_prime();
            let rug_prime = rug_a.is_probably_prime(25) != rug::integer::IsPrime::No;
            if arbi_a.to_u64().is_some() {
                assert_eq!(arbi_prime, rug_prime);
            }
        }
        12 => {
            if arbi_b.is_odd() && let Some(arbi) = arbi_a.jacobi_symbol(arbi_b) {
                let rug = rug_a.jacobi(rug_b);
                assert_eq!(i32::from(arbi), rug);
            }
        }
        _ => {
            let is_pow2 = arbi_a.is_power_of_two();
            assert_eq!(is_pow2, arbi_a.count_ones() == 1);
            if let Some(next_pow2) = arbi_a.checked_next_power_of_two() {
                assert!(next_pow2 >= *arbi_a);
                assert!(next_pow2.is_power_of_two() || next_pow2 == ArbiUint::zero());
            }
        }
    }
}
