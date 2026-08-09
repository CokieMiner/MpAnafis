//! Arithmetic fuzzing checks for `MpUint`.

use mp_anafis::MpUint;
use rug::Integer;

pub fn fuzz_all(mp_a: &MpUint, mp_b: &MpUint, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 11 {
        0 => {
            let mp = mp_a.clone() + mp_b.clone();
            let rug = rug_a.clone() + rug_b.clone();
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        1 => {
            if mp_a >= mp_b {
                let mp = mp_a.clone() - mp_b.clone();
                let rug = rug_a.clone() - rug_b.clone();
                assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
            }
        }
        2 => {
            let mp = mp_a.clone() * mp_b.clone();
            let rug = rug_a.clone() * rug_b.clone();
            assert_eq!(format!("{:x}", mp), format!("{:x}", rug));
        }
        3 => {
            if *rug_b != 0 {
                let mp_div = mp_a.clone() / mp_b.clone();
                let rug_div = rug_a.clone() / rug_b.clone();
                assert_eq!(format!("{:x}", mp_div), format!("{:x}", rug_div));

                let mp_rem = mp_a.clone() % mp_b.clone();
                let rug_rem = rug_a.clone() % rug_b.clone();
                assert_eq!(format!("{:x}", mp_rem), format!("{:x}", rug_rem));
            }
        }
        4 => {
            assert_eq!(mp_a.strict_add(mp_b), mp_a.clone() + mp_b.clone());
            assert_eq!(mp_a.saturating_add(mp_b), mp_a.clone() + mp_b.clone());
            assert_eq!(mp_a.wrapping_add(mp_b), mp_a.clone() + mp_b.clone());
        }
        5 => {
            if mp_a >= mp_b {
                assert_eq!(mp_a.strict_sub(mp_b), mp_a.clone() - mp_b.clone());
                assert_eq!(mp_a.saturating_sub(mp_b), mp_a.clone() - mp_b.clone());
                assert_eq!(mp_a.wrapping_sub(mp_b), mp_a.clone() - mp_b.clone());
            } else {
                assert_eq!(mp_a.saturating_sub(mp_b), MpUint::zero());
            }
        }
        6 => {
            assert_eq!(mp_a.strict_mul(mp_b), mp_a.clone() * mp_b.clone());
            assert_eq!(mp_a.saturating_mul(mp_b), mp_a.clone() * mp_b.clone());
            assert_eq!(mp_a.wrapping_mul(mp_b), mp_a.clone() * mp_b.clone());
        }
        7 => {
            if *rug_b != 0 {
                assert_eq!(mp_a.strict_div(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.saturating_div(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.wrapping_div(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.strict_rem(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.saturating_rem(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.wrapping_rem(mp_b), mp_a.clone() % mp_b.clone());
            }
        }
        8 => {
            let (sum, _ov) = mp_a.overflowing_add(mp_b);
            assert_eq!(sum, mp_a.clone() + mp_b.clone());
            let (prod, _ov) = mp_a.overflowing_mul(mp_b);
            assert_eq!(prod, mp_a.clone() * mp_b.clone());
            if mp_a >= mp_b {
                let (diff, _ov) = mp_a.overflowing_sub(mp_b);
                assert_eq!(diff, mp_a.clone() - mp_b.clone());
            }
            if *rug_b != 0 {
                let (div, _ov) = mp_a.overflowing_div(mp_b);
                assert_eq!(div, mp_a.clone() / mp_b.clone());
                let (rem, _ov) = mp_a.overflowing_rem(mp_b);
                assert_eq!(rem, mp_a.clone() % mp_b.clone());
            }
        }
        9 => {
            let mut acc = mp_a.clone();
            acc.assign_add(mp_a, mp_b);
            assert_eq!(acc, mp_a.clone() + mp_b.clone());
            if mp_a >= mp_b {
                let mut diff = mp_a.clone();
                let _ = diff.assign_sub(mp_a, mp_b);
                assert_eq!(diff, mp_a.clone() - mp_b.clone());
            }
        }
        10 => {
            assert_eq!(mp_a.checked_add(mp_b), Some(mp_a.clone() + mp_b.clone()));
            assert_eq!(mp_a.try_add(mp_b).ok(), Some(mp_a.clone() + mp_b.clone()));
            assert_eq!(mp_a.checked_mul(mp_b), Some(mp_a.clone() * mp_b.clone()));
            assert_eq!(mp_a.try_mul(mp_b).ok(), Some(mp_a.clone() * mp_b.clone()));
            if mp_a >= mp_b {
                assert_eq!(mp_a.checked_sub(mp_b), Some(mp_a.clone() - mp_b.clone()));
                assert_eq!(mp_a.try_sub(mp_b).ok(), Some(mp_a.clone() - mp_b.clone()));
            } else {
                assert_eq!(mp_a.checked_sub(mp_b), None);
                assert!(mp_a.try_sub(mp_b).is_err());
            }
            if *rug_b != 0 {
                assert_eq!(mp_a.checked_div(mp_b), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.try_div(mp_b).ok(), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.checked_rem(mp_b), Some(mp_a.clone() % mp_b.clone()));
                assert_eq!(mp_a.try_rem(mp_b).ok(), Some(mp_a.clone() % mp_b.clone()));
                if let Some((q, r)) = mp_a.div_rem(mp_b) {
                    assert_eq!(q, mp_a.clone() / mp_b.clone());
                    assert_eq!(r, mp_a.clone() % mp_b.clone());
                }
            } else {
                assert_eq!(mp_a.checked_div(mp_b), None);
                assert!(mp_a.try_div(mp_b).is_err());
                assert_eq!(mp_a.checked_rem(mp_b), None);
                assert!(mp_a.try_rem(mp_b).is_err());
                assert_eq!(mp_a.div_rem(mp_b), None);
            }
        }
        _ => {
            let fma = mp_a.mul_add(mp_a, mp_b);
            assert_eq!(fma, mp_a.clone() + mp_a.clone() * mp_b.clone());
            let mid = mp_a.midpoint(mp_b);
            assert!(mid <= mp_a.clone().max(mp_b.clone()));
            let diff = mp_a.abs_diff(mp_b);
            if mp_a >= mp_b {
                assert_eq!(diff, mp_a.clone() - mp_b.clone());
            } else {
                assert_eq!(diff, mp_b.clone() - mp_a.clone());
            }
        }
    }
}
