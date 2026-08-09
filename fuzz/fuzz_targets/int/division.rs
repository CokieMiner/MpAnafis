//! Division modes and divisibility fuzzing checks for `MpInt`.

use mp_anafis::MpInt;
use rug::{Integer, ops::DivRounding};

pub fn fuzz_all(
    mp_a: &MpInt,
    mp_b: &MpInt,
    rug_a: &Integer,
    rug_b: &Integer,
    selector: u8,
) {
    match selector % 3 {
        0 => {
            if *rug_b != 0 {
                assert_eq!(mp_a.div_trunc(mp_b), mp_a.clone() / mp_b.clone());
                assert_eq!(mp_a.rem_trunc(mp_b), mp_a.clone() % mp_b.clone());
                assert_eq!(mp_a.checked_div_trunc(mp_b), Some(mp_a.clone() / mp_b.clone()));
                assert_eq!(mp_a.checked_rem_trunc(mp_b), Some(mp_a.clone() % mp_b.clone()));
                let (rug_q, rug_r) = rug_a.clone().div_rem_euc(rug_b.clone());
                assert_eq!(format!("{}", mp_a.div_euclid(mp_b)), rug_q.to_string());
                assert_eq!(format!("{}", mp_a.rem_euclid(mp_b)), rug_r.to_string());
                assert_eq!(
                    mp_a.checked_div_euclid(mp_b).map(|value| value.to_string()),
                    Some(rug_q.to_string())
                );
                assert_eq!(
                    mp_a.checked_rem_euclid(mp_b).map(|value| value.to_string()),
                    Some(rug_r.to_string())
                );
                if let Some((q, r)) = mp_a.div_rem_euclid(mp_b) {
                    assert_eq!(q.to_string(), rug_q.to_string());
                    assert_eq!(r.to_string(), rug_r.to_string());
                }
            } else {
                assert_eq!(mp_a.checked_div_trunc(mp_b), None);
                assert_eq!(mp_a.checked_rem_trunc(mp_b), None);
                assert_eq!(mp_a.checked_div_euclid(mp_b), None);
                assert_eq!(mp_a.checked_rem_euclid(mp_b), None);
                assert_eq!(mp_a.div_rem_euclid(mp_b), None);
            }
        }
        1 => {
            if *rug_b != 0 {
                let (rug_q, rug_r) = rug_a.clone().div_rem_floor(rug_b.clone());
                assert_eq!(mp_a.div_floor(mp_b).to_string(), rug_q.to_string());
                assert_eq!(mp_a.mod_floor(mp_b).to_string(), rug_r.to_string());
                assert_eq!(
                    mp_a.checked_div_floor(mp_b).map(|value| value.to_string()),
                    Some(rug_q.to_string())
                );
                assert_eq!(
                    mp_a.checked_mod_floor(mp_b).map(|value| value.to_string()),
                    Some(rug_r.to_string())
                );
                if let Some((q, r)) = mp_a.div_rem_floor(mp_b) {
                    assert_eq!(q.to_string(), rug_q.to_string());
                    assert_eq!(r.to_string(), rug_r.to_string());
                }
                let ceil = mp_a.div_ceil(mp_b);
                let rug_ceil = rug_a.clone().div_ceil(rug_b);
                assert_eq!(ceil.to_string(), rug_ceil.to_string());
                assert_eq!(mp_a.checked_div_ceil(mp_b), Some(ceil.clone()));
            } else {
                assert_eq!(mp_a.checked_div_floor(mp_b), None);
                assert_eq!(mp_a.checked_mod_floor(mp_b), None);
                assert_eq!(mp_a.div_rem_floor(mp_b), None);
                assert_eq!(mp_a.checked_div_ceil(mp_b), None);
            }
        }
        _ => {
            if *rug_b != 0 {
                let div = mp_a.is_divisible_by(mp_b);
                assert_eq!(div, (mp_a.clone() % mp_b.clone()) == MpInt::zero());
                let div_of = mp_b.is_divisor_of(mp_a);
                assert_eq!(div_of, div);
            }
        }
    }
}
