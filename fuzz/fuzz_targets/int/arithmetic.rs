//! Arithmetic, sign, and checked/saturating/wrapping checks for `MpInt`.

use mp_anafis::MpInt;
use rug::Integer;

pub fn fuzz_all(mp_a: &MpInt, mp_b: &MpInt, rug_a: &Integer, rug_b: &Integer, selector: u8) {
    match selector % 11 {
        0 => {
            let mp = mp_a.clone() + mp_b.clone();
            let rug = rug_a.clone() + rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        1 => {
            let mp = mp_a.clone() - mp_b.clone();
            let rug = rug_a.clone() - rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        2 => {
            let mp = mp_a.clone() * mp_b.clone();
            let rug = rug_a.clone() * rug_b.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        3 => {
            if *rug_b != 0 {
                let mp_div = mp_a.clone() / mp_b.clone();
                let rug_div = rug_a.clone() / rug_b.clone();
                assert_eq!(format!("{mp_div:x}"), format!("{rug_div:x}"));

                let mp_rem = mp_a.clone() % mp_b.clone();
                let rug_rem = rug_a.clone() % rug_b.clone();
                assert_eq!(format!("{mp_rem:x}"), format!("{rug_rem:x}"));
            }
        }
        4 => {
            let mp = -mp_a.clone();
            let rug = -rug_a.clone();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
        }
        5 => {
            let mp = mp_a.abs();
            let rug = rug_a.clone().abs();
            assert_eq!(format!("{mp:x}"), format!("{rug:x}"));
            assert_eq!(mp_a.checked_abs(), Some(mp));
        }
        6 => {
            assert_eq!(mp_a.saturating_add(mp_b), mp_a.clone() + mp_b.clone());
            assert_eq!(mp_a.wrapping_add(mp_b), mp_a.clone() + mp_b.clone());
            assert_eq!(mp_a.saturating_sub(mp_b), mp_a.clone() - mp_b.clone());
            assert_eq!(mp_a.wrapping_sub(mp_b), mp_a.clone() - mp_b.clone());
            assert_eq!(mp_a.saturating_mul(mp_b), mp_a.clone() * mp_b.clone());
            assert_eq!(mp_a.wrapping_mul(mp_b), mp_a.clone() * mp_b.clone());
        }
        7 => {
            if *rug_b != 0 {
                assert_eq!(mp_a.saturating_div(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.wrapping_div(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.saturating_rem(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.wrapping_rem(mp_b), mp_a.clone() % mp_b.clone());
            }
        }
        8 => {
            let (sum, _ov) = mp_a.overflowing_add(mp_b);
            assert_eq!(sum, mp_a.clone() + mp_b.clone());
            let (diff, _ov) = mp_a.overflowing_sub(mp_b);
            assert_eq!(diff, mp_a.clone() - mp_b.clone());
            let (prod, _ov) = mp_a.overflowing_mul(mp_b);
            assert_eq!(prod, mp_a.clone() * mp_b.clone());
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
            let mut diff = mp_a.clone();
            diff.assign_sub(mp_a, mp_b);
            assert_eq!(diff, mp_a.clone() - mp_b.clone());
        }
        10 => {
            assert_eq!(mp_a.checked_add(mp_b), Some(mp_a.clone() + mp_b.clone()));
            assert_eq!(mp_a.try_add(mp_b).ok(), Some(mp_a.clone() + mp_b.clone()));
            assert_eq!(mp_a.checked_sub(mp_b), Some(mp_a.clone() - mp_b.clone()));
            assert_eq!(mp_a.try_sub(mp_b).ok(), Some(mp_a.clone() - mp_b.clone()));
            assert_eq!(mp_a.checked_mul(mp_b), Some(mp_a.clone() * mp_b.clone()));
            assert_eq!(mp_a.try_mul(mp_b).ok(), Some(mp_a.clone() * mp_b.clone()));
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
            let avg = (mp_a.clone() + mp_b.clone()) / MpInt::from(2);
            assert_eq!(mid, avg);
            let abs_d = mp_a.abs_diff(mp_b);
            let diff = (mp_a.clone() - mp_b.clone()).abs();
            assert_eq!(format!("{abs_d:x}"), format!("{diff:x}"));

            let sig = mp_a.signum();
            if *mp_a > MpInt::zero() {
                assert_eq!(sig, MpInt::one());
            } else if *mp_a < MpInt::zero() {
                assert_eq!(sig, MpInt::minus_one());
            } else {
                assert_eq!(sig, MpInt::zero());
            }
            assert_eq!(mp_a.is_positive(), *mp_a > MpInt::zero());
            assert_eq!(mp_a.is_negative(), *mp_a < MpInt::zero());
            assert_eq!(mp_a.is_zero(), *mp_a == MpInt::zero());
            assert_eq!(mp_a.is_even(), (mp_a.clone() % MpInt::from(2)) == MpInt::zero());
            assert_eq!(mp_a.is_odd(), !mp_a.is_even());
        }
    }
}
