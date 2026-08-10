//! Arithmetic fuzzing checks for `ArbiUint`.

use arbi_anafis::ArbiUint;
use rug::Integer;

pub fn fuzz_all(arbi_a: &ArbiUint, arbi_b: &ArbiUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 11 {
        0 => {
            let arbi = arbi_a.clone() + arbi_b.clone();
            let rug = rug_a.clone() + rug_b.clone();
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        1 => {
            if arbi_a >= arbi_b {
                let arbi = arbi_a.clone() - arbi_b.clone();
                let rug = rug_a.clone() - rug_b.clone();
                assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
            }
        }
        2 => {
            let arbi = arbi_a.clone() * arbi_b.clone();
            let rug = rug_a.clone() * rug_b.clone();
            assert_eq!(format!("{:x}", arbi), format!("{:x}", rug));
        }
        3 => {
            if *rug_b != 0 {
                let arbi_div = arbi_a.clone() / arbi_b.clone();
                let rug_div = rug_a.clone() / rug_b.clone();
                assert_eq!(format!("{:x}", arbi_div), format!("{:x}", rug_div));

                let arbi_rem = arbi_a.clone() % arbi_b.clone();
                let rug_rem = rug_a.clone() % rug_b.clone();
                assert_eq!(format!("{:x}", arbi_rem), format!("{:x}", rug_rem));
            }
        }
        4 => {
            assert_eq!(arbi_a.strict_add(arbi_b), arbi_a.clone() + arbi_b.clone());
            assert_eq!(arbi_a.saturating_add(arbi_b), arbi_a.clone() + arbi_b.clone());
            assert_eq!(arbi_a.wrapping_add(arbi_b), arbi_a.clone() + arbi_b.clone());
        }
        5 => {
            if arbi_a >= arbi_b {
                assert_eq!(arbi_a.strict_sub(arbi_b), arbi_a.clone() - arbi_b.clone());
                assert_eq!(arbi_a.saturating_sub(arbi_b), arbi_a.clone() - arbi_b.clone());
                assert_eq!(arbi_a.wrapping_sub(arbi_b), arbi_a.clone() - arbi_b.clone());
            } else {
                assert_eq!(arbi_a.saturating_sub(arbi_b), ArbiUint::zero());
            }
        }
        6 => {
            assert_eq!(arbi_a.strict_mul(arbi_b), arbi_a.clone() * arbi_b.clone());
            assert_eq!(arbi_a.saturating_mul(arbi_b), arbi_a.clone() * arbi_b.clone());
            assert_eq!(arbi_a.wrapping_mul(arbi_b), arbi_a.clone() * arbi_b.clone());
        }
        7 => {
            if *rug_b != 0 {
                assert_eq!(arbi_a.strict_div(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.saturating_div(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.wrapping_div(arbi_b), arbi_a.clone() / arbi_b.clone());
                assert_eq!(arbi_a.strict_rem(arbi_b), arbi_a.clone() % arbi_b.clone());
                assert_eq!(arbi_a.saturating_rem(arbi_b), arbi_a.clone() % arbi_b.clone());
                assert_eq!(arbi_a.wrapping_rem(arbi_b), arbi_a.clone() % arbi_b.clone());
            }
        }
        8 => {
            let (sum, _ov) = arbi_a.overflowing_add(arbi_b);
            assert_eq!(sum, arbi_a.clone() + arbi_b.clone());
            let (prod, _ov) = arbi_a.overflowing_mul(arbi_b);
            assert_eq!(prod, arbi_a.clone() * arbi_b.clone());
            if arbi_a >= arbi_b {
                let (diff, _ov) = arbi_a.overflowing_sub(arbi_b);
                assert_eq!(diff, arbi_a.clone() - arbi_b.clone());
            }
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
            if arbi_a >= arbi_b {
                let mut diff = arbi_a.clone();
                let _ = diff.assign_sub(arbi_a, arbi_b);
                assert_eq!(diff, arbi_a.clone() - arbi_b.clone());
            }
        }
        10 => {
            assert_eq!(arbi_a.checked_add(arbi_b), Some(arbi_a.clone() + arbi_b.clone()));
            assert_eq!(arbi_a.try_add(arbi_b).ok(), Some(arbi_a.clone() + arbi_b.clone()));
            assert_eq!(arbi_a.checked_mul(arbi_b), Some(arbi_a.clone() * arbi_b.clone()));
            assert_eq!(arbi_a.try_mul(arbi_b).ok(), Some(arbi_a.clone() * arbi_b.clone()));
            if arbi_a >= arbi_b {
                assert_eq!(arbi_a.checked_sub(arbi_b), Some(arbi_a.clone() - arbi_b.clone()));
                assert_eq!(arbi_a.try_sub(arbi_b).ok(), Some(arbi_a.clone() - arbi_b.clone()));
            } else {
                assert_eq!(arbi_a.checked_sub(arbi_b), None);
                assert!(arbi_a.try_sub(arbi_b).is_err());
            }
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
            assert!(mid <= arbi_a.clone().max(arbi_b.clone()));
            let diff = arbi_a.abs_diff(arbi_b);
            if arbi_a >= arbi_b {
                assert_eq!(diff, arbi_a.clone() - arbi_b.clone());
            } else {
                assert_eq!(diff, arbi_b.clone() - arbi_a.clone());
            }
        }
    }
}
