//! Number theory, powers, and modular arithmetic checks for `MpUint`.

use mp_anafis::MpUint;
use rug::{Integer, ops::Pow};

pub fn fuzz_all(mp_a: &MpUint, mp_b: &MpUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 14 {
        0 => {
            if let Some((g, l)) = mp_a.gcd_lcm(mp_b) {
                let rug_g = rug_a.clone().gcd(rug_b);
                let rug_l = rug_a.clone().lcm(rug_b);
                assert_eq!(format!("{:x}", g), format!("{:x}", rug_g));
                assert_eq!(format!("{:x}", l), format!("{:x}", rug_l));
                assert_eq!(mp_a.gcd(mp_b), g);
                if let Some(l_only) = mp_a.lcm(mp_b) {
                    assert_eq!(l_only, l);
                }
                assert_eq!(mp_a.is_coprime(mp_b), g == MpUint::one());
            }
        }
        1 => {
            if *rug_b != 0 && let Some((g, _x, _y)) = mp_a.extended_gcd(mp_b) {
                assert_eq!(g, mp_a.gcd(mp_b));
            }
        }
        2 => {
            if let Some(root) = mp_a.isqrt() {
                let root_sq = root.clone() * root.clone();
                let next_sq = (root.clone() + MpUint::one()) * (root.clone() + MpUint::one());
                assert!(root_sq <= *mp_a);
                assert!(*mp_a <= next_sq || *mp_a == MpUint::zero());
                assert_eq!(mp_a.is_perfect_square(), root_sq == *mp_a);
            }
        }
        3 => {
            let n = (mp_b.to_u64().unwrap_or(0) % 10) as u32;
            if n > 0 && let Some(root) = mp_a.nth_root(n) {
                let root_pow = root.clone().pow(n);
                assert!(root_pow <= *mp_a);
            }
        }
        4 => {
            if let Some((s, r)) = mp_a.sqrt_rem() {
                let sq = s.clone() * s.clone() + r;
                assert_eq!(sq, *mp_a);
            }
        }
        5 => {
            let exp = (mp_b.to_u64().unwrap_or(0) % 16) as u32;
            let mp = mp_a.pow(exp);
            let rug = Integer::from(rug_a.pow(exp));
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            assert_eq!(mp_a.checked_pow(exp), Some(mp));
        }
        6 => {
            if *rug_b > 1 && let Some(mp) = mp_a.add_mod(mp_b, mp_b) {
                let rug = Integer::from(rug_a + rug_b) % rug_b;
                assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            }
        }
        7 => {
            if *rug_b > 1 && let Some(mp) = mp_a.mul_mod(mp_b, mp_b) {
                let rug = Integer::from(rug_a * rug_b) % rug_b;
                assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            }
        }
        8 => {
            if *rug_b > 1 {
                let exp = mp_a.clone() % MpUint::from(256_u32);
                let rug_exp = rug_a.clone() % 256;
                if let Some(mp) = mp_a.pow_mod(&exp, mp_b) {
                    let rug = match rug_a.clone().pow_mod(&rug_exp, rug_b) {
                        Ok(val) => val,
                        Err(_) => unreachable!(),
                    };
                    assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
                }
            }
        }
        9 => {
            if *rug_b > 1
                && let Some(mp) = mp_a.invert(mp_b)
                && let Ok(rug) = rug_a.clone().invert(rug_b)
            {
                assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            }
        }
        10 => {
            if *rug_b > 1 && let Some(mp) = mp_a.sub_mod(mp_a, mp_b) {
                let mut rug = Integer::from(0);
                if rug < 0 {
                    rug += rug_b;
                }
                rug %= rug_b;
                assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            }
        }
        11 => {
            let mp_prime = mp_a.is_prime();
            let rug_prime = rug_a.is_probably_prime(25) != rug::integer::IsPrime::No;
            if mp_a.to_u64().is_some() {
                assert_eq!(mp_prime, rug_prime);
            }
        }
        12 => {
            if mp_b.is_odd() && let Some(mp) = mp_a.jacobi_symbol(mp_b) {
                let rug = rug_a.jacobi(rug_b);
                assert_eq!(i32::from(mp), rug);
            }
        }
        _ => {
            let is_pow2 = mp_a.is_power_of_two();
            assert_eq!(is_pow2, mp_a.count_ones() == 1);
            if let Some(next_pow2) = mp_a.checked_next_power_of_two() {
                assert!(next_pow2 >= *mp_a);
                assert!(next_pow2.is_power_of_two() || next_pow2 == MpUint::zero());
            }
        }
    }
}
