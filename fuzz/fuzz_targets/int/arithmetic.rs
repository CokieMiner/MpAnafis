//! Arithmetic, sign, and checked/saturating/wrapping checks for `ArbiInt`.

use arbi_anafis::ArbiInt;
use rug::Integer;

pub fn fuzz_all(arbi_a: &ArbiInt, arbi_b: &ArbiInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 11 {
        0 => {
            let arbi = arbi_a.clone() + arbi_b.clone();
            let rug = rug_a.clone() + rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        1 => {
            let arbi = arbi_a.clone() - arbi_b.clone();
            let rug = rug_a.clone() - rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        2 => {
            let arbi = arbi_a.clone() * arbi_b.clone();
            let rug = rug_a.clone() * rug_b.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        3 => {
            if *rug_b != 0 {
                let arbi_div = arbi_a.clone() / arbi_b.clone();
                let rug_div = rug_a.clone() / rug_b.clone();
                assert_eq!(format!("{arbi_div:x}"), format!("{rug_div:x}"));

                let arbi_rem = arbi_a.clone() % arbi_b.clone();
                let rug_rem = rug_a.clone() % rug_b.clone();
                assert_eq!(format!("{arbi_rem:x}"), format!("{rug_rem:x}"));
            }
        }
        4 => {
            let arbi = -arbi_a.clone();
            let rug = -rug_a.clone();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
        }
        5 => {
            let arbi = arbi_a.abs();
            let rug = rug_a.clone().abs();
            assert_eq!(format!("{arbi:x}"), format!("{rug:x}"));
            assert_eq!(arbi_a.checked_abs(), Some(arbi));
        }
        6 => {
            assert_eq!(arbi_a.saturating_add(arbi_b), arbi_a.clone() + arbi_b.clone());
            assert_eq!(arbi_a.wrapping_add(arbi_b), arbi_a.clone() + arbi_b.clone());
            assert_eq!(arbi_a.saturating_sub(arbi_b), arbi_a.clone() - arbi_b.clone());
            assert_eq!(arbi_a.wrapping_sub(arbi_b), arbi_a.clone() - arbi_b.clone());
            assert_eq!(arbi_a.saturating_mul(arbi_b), arbi_a.clone() * arbi_b.clone());
            assert_eq!(arbi_a.wrapping_mul(arbi_b), arbi_a.clone() * arbi_b.clone());
        }
        7 => {
            if *rug_b != 0 {
                assert_eq!(arbi_a.saturating_div(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.wrapping_div(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.saturating_rem(arbi_b), arbi_a.clone() % arbi_b.clone());
                assert_eq!(arbi_a.wrapping_rem(arbi_b), arbi_a.clone() % arbi_b.clone());
            }
        }
        8 => {
            let (sum, _ov) = arbi_a.overflowing_add(arbi_b);
            assert_eq!(sum, arbi_a.clone() + arbi_b.clone());
            let (diff, _ov) = arbi_a.overflowing_sub(arbi_b);
            assert_eq!(diff, arbi_a.clone() - arbi_b.clone());
            let (prod, _ov) = arbi_a.overflowing_mul(arbi_b);
            assert_eq!(prod, arbi_a.clone() * arbi_b.clone());
            if *rug_b != 0 {
                let (div, _ov) = arbi_a.overflowing_div(arbi_b);
                assert_eq!(div, arbi_a.clone() / arbi_b.clone());
                let (rem, _ov) = arbi_a.overflowing_rem(arbi_b);
                assert_eq!(rem, arbi_a.clone() % arbi_b.clone());
            }
        }
        9 => {
            let mut acc = arbi_a.clone();
            acc.assign_add(arbi_a, arbi_b);
            assert_eq!(acc, arbi_a.clone() + arbi_b.clone());
            let mut diff = arbi_a.clone();
            diff.assign_sub(arbi_a, arbi_b);
            assert_eq!(diff, arbi_a.clone() - arbi_b.clone());
        }
        10 => {
            assert_eq!(arbi_a.checked_add(arbi_b), Some(arbi_a.clone() + arbi_b.clone()));
            assert_eq!(arbi_a.try_add(arbi_b).ok(), Some(arbi_a.clone() + arbi_b.clone()));
            assert_eq!(arbi_a.checked_sub(arbi_b), Some(arbi_a.clone() - arbi_b.clone()));
            assert_eq!(arbi_a.try_sub(arbi_b).ok(), Some(arbi_a.clone() - arbi_b.clone()));
            assert_eq!(arbi_a.checked_mul(arbi_b), Some(arbi_a.clone() * arbi_b.clone()));
            assert_eq!(arbi_a.try_mul(arbi_b).ok(), Some(arbi_a.clone() * arbi_b.clone()));
            if *rug_b != 0 {
                assert_eq!(arbi_a.checked_div(arbi_b), Some(arbi_a.clone() / arbi_b.clone()));
                assert_eq!(arbi_a.try_div(arbi_b).ok(), Some(arbi_a.clone() / arbi_b.clone()));
                assert_eq!(arbi_a.checked_rem(arbi_b), Some(arbi_a.clone() % arbi_b.clone()));
                assert_eq!(arbi_a.try_rem(arbi_b).ok(), Some(arbi_a.clone() % arbi_b.clone()));
                if let Some((q, r)) = arbi_a.div_rem(arbi_b) {
                    assert_eq!(q, arbi_a.clone() / arbi_b.clone());
                    assert_eq!(r, arbi_a.clone() % arbi_b.clone());
                }
            } else {
                assert_eq!(arbi_a.checked_div(arbi_b), None);
                assert!(arbi_a.try_div(arbi_b).is_err());
                assert_eq!(arbi_a.checked_rem(arbi_b), None);
                assert!(arbi_a.try_rem(arbi_b).is_err());
                assert_eq!(arbi_a.div_rem(arbi_b), None);
            }
        }
        _ => {
            let fma = arbi_a.mul_add(arbi_a, arbi_b);
            assert_eq!(fma, arbi_a.clone() + arbi_a.clone() * arbi_b.clone());
            let mid = arbi_a.midpoint(arbi_b);
            let avg = (arbi_a.clone() + arbi_b.clone()) / ArbiInt::from(2);
            assert_eq!(mid, avg);
            let abs_d = arbi_a.abs_diff(arbi_b);
            let diff = (arbi_a.clone() - arbi_b.clone()).abs();
            assert_eq!(format!("{abs_d:x}"), format!("{diff:x}"));

            let sig = arbi_a.signum();
            if *arbi_a > ArbiInt::zero() {
                assert_eq!(sig, ArbiInt::one());
            } else if *arbi_a < ArbiInt::zero() {
                assert_eq!(sig, ArbiInt::minus_one());
            } else {
                assert_eq!(sig, ArbiInt::zero());
            }
            assert_eq!(arbi_a.is_positive(), *arbi_a > ArbiInt::zero());
            assert_eq!(arbi_a.is_negative(), *arbi_a < ArbiInt::zero());
            assert_eq!(arbi_a.is_zero(), *arbi_a == ArbiInt::zero());
            assert_eq!(arbi_a.is_even(), (arbi_a.clone() % ArbiInt::from(2)) == ArbiInt::zero());
            assert_eq!(arbi_a.is_odd(), !arbi_a.is_even());
        }
    }
}
